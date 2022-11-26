import numpy as np
import re
from typing import Dict
import warnings

from pyRDDLGym.Core.Debug.decompiler import RDDLDecompiler
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLActionPreconditionNotSatisfiedError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidActionError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidNumberOfArgumentsError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLInvalidObjectError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLNotImplementedError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLStateInvariantNotSatisfiedError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLTypeError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLUndefinedVariableError
from pyRDDLGym.Core.ErrorHandling.RDDLException import RDDLValueOutOfRangeError
from pyRDDLGym.Core.Parser.expr import Expression
from pyRDDLGym.Core.Parser.rddl import RDDL
from pyRDDLGym.Core.Simulator.LiftedRDDLStaticAnalysis import LiftedRDDLStaticAnalysis


class LiftedRDDLSimulator:
    
    INT = np.int32
    REAL = np.float64
        
    NUMPY_TYPES = {
        'int': INT,
        'real': REAL,
        'bool': bool
    }
    
    DEFAULT_VALUES = {
        'int': 0,
        'real': 0.0,
        'bool': False
    }
    
    VALID_SYMBOLS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    def __init__(self,
                 rddl: RDDL,
                 allow_synchronous_state: bool=True,
                 rng: np.random.Generator=np.random.default_rng(),
                 debug: bool=False) -> None:
        '''Creates a new simulator for the given RDDL model.
        
        :param rddl: the RDDL model
        :param allow_synchronous_state: whether state-fluent can be synchronous
        :param rng: the random number generator
        :param debug: print compiler information
        '''
        self.rddl = rddl
        self.rng = rng
        self.debug = debug
        
        self.domain = rddl.domain
        self.instance = rddl.instance
        self.non_fluents = rddl.non_fluents
        
        self.static = LiftedRDDLStaticAnalysis(rddl, allow_synchronous_state)
        self.levels = self.static.compute_levels()
        self.next_states = self.static.next_states
        
        self._compile()
        self.action_pattern = re.compile(r'(\S*?)\((\S.*?)\)', re.VERBOSE)
        
        # basic operations
        self.ARITHMETIC_OPS = {
            '+': np.add,
            '-': np.subtract,
            '*': np.multiply,
            '/': np.divide
        }    
        self.RELATIONAL_OPS = {
            '>=': np.greater_equal,
            '<=': np.less_equal,
            '<': np.less,
            '>': np.greater,
            '==': np.equal,
            '~=': np.not_equal
        }
        self.LOGICAL_OPS = {
            '^': np.logical_and,
            '|': np.logical_or,
            '~': np.logical_xor,
            '=>': lambda e1, e2: np.logical_or(np.logical_not(e1), e2),
            '<=>': np.equal
        }
        self.AGGREGATION_OPS = {
            'sum': np.sum,
            'avg': np.mean,
            'prod': np.prod,
            'min': np.min,
            'max': np.max,
            'forall': np.all,
            'exists': np.any  
        }
        self.UNARY = {        
            'abs': np.abs,
            'sgn': np.sign,
            'round': np.round,
            'floor': np.floor,
            'ceil': np.ceil,
            'cos': np.cos,
            'sin': np.sin,
            'tan': np.tan,
            'acos': np.arccos,
            'asin': np.arcsin,
            'atan': np.arctan,
            'cosh': np.cosh,
            'sinh': np.sinh,
            'tanh': np.tanh,
            'exp': np.exp,
            'ln': np.log,
            'sqrt': np.sqrt
        }        
        self.BINARY = {
            'div': np.floor_divide,
            'mod': np.mod,
            'min': np.minimum,
            'max': np.maximum,
            'pow': np.power,
            'log': lambda x, y: np.log(x) / np.log(y)
        }
    
    # ===========================================================================
    # compilation time
    # ===========================================================================
    
    def _compile(self):
        self._compile_types()
        self._compile_objects()
        self._compile_initial_values()
        
    def _compile_types(self):
        self.pvar_types = {}
        for pvar in self.domain.pvariables:
            primed_name = name = pvar.name
            if pvar.is_state_fluent():
                primed_name = name + '\''
            ptypes = pvar.param_types
            if ptypes is None:
                ptypes = []
            self.pvar_types[name] = ptypes
            self.pvar_types[primed_name] = ptypes
            
        self.cpf_types = {}
        for cpf in self.domain.cpfs[1]:
            _, (name, objects) = cpf.pvar
            if objects is None:
                objects = [] 
            types = self.pvar_types[name]
            self.cpf_types[name] = [(o, types[i]) for i, o in enumerate(objects)]
    
    def _compile_objects(self): 
        self.objects, self.objects_to_index = {}, {}     
        for name, ptype in self.non_fluents.objects:
            indices = {obj: index for index, obj in enumerate(ptype)}
            self.objects[name] = indices
            overlap = indices.keys() & self.objects_to_index.keys()
            if overlap:
                raise RDDLInvalidObjectError(
                    'Multiple types share the same object <{}>.'.format(overlap))
            self.objects_to_index.update(indices)
        
    def _objects_to_coordinates(self, objects, msg):
        try:
            return tuple(self.objects_to_index[obj] for obj in objects)
        except:
            for obj in objects:
                if obj not in self.objects_to_index:
                    raise RDDLInvalidObjectError(
                        'Object <{}> declared in {} is not valid.'.format(obj, msg))
        
    def _compile_initial_values(self):
        
        # get default values from domain
        self.init_values = {}
        self.noop_actions = {}
        for pvar in self.domain.pvariables:
            name = pvar.name                             
            ptypes = pvar.param_types
            prange = pvar.range
            
            value = pvar.default
            if value is None:
                value = LiftedRDDLSimulator.DEFAULT_VALUES[prange]
            if ptypes is None:
                self.init_values[name] = value              
            else: 
                self.init_values[name] = np.full(
                    shape=tuple(len(self.objects[obj]) for obj in ptypes),
                    fill_value=value,
                    dtype=LiftedRDDLSimulator.NUMPY_TYPES[prange])
            
            if pvar.is_action_fluent():
                self.noop_actions[name] = self.init_values[name]
        
        # override default values with instance
        if hasattr(self.instance, 'init_state'):
            for (name, objects), value in self.instance.init_state:
                if objects is not None:
                    coords = self._objects_to_coordinates(objects, 'init-state')
                    self.init_values[name][coords] = value   
        
        if hasattr(self.non_fluents, 'init_non_fluent'):
            for (name, objects), value in self.non_fluents.init_non_fluent:
                if objects is not None:
                    coords = self._objects_to_coordinates(objects, 'non-fluents')
                    self.init_values[name][coords] = value
    
    # ===========================================================================
    # error checks
    # ===========================================================================
    
    @staticmethod
    def _print_stack_trace(expr):
        if isinstance(expr, Expression):
            trace = RDDLDecompiler().decompile_expr(expr)
        else:
            trace = str(expr)
        return '>> ' + trace
    
    @staticmethod
    def _check_type(value, valid, msg, expr):
        value = value.dtype
        if value != valid:
            raise RDDLTypeError(
                '{} must evaluate to {}, got {}.'.format(msg, valid, value) + 
                '\n' + LiftedRDDLSimulator._print_stack_trace(expr))
    
    @staticmethod
    def _check_types(value1, value2, valid, msg, expr):
        value1 = value1.dtype
        value2 = value2.dtype
        if value1 != valid and value2 != valid:
            raise RDDLTypeError(
                '{} must evaluate to {}, got {} and {}.'.format(
                    msg, valid, value1, value2) + 
                '\n' + LiftedRDDLSimulator._print_stack_trace(expr))
            
    @staticmethod
    def _check_op(op, valid, msg, expr):
        if op not in valid:
            raise RDDLNotImplementedError(
                '{} operator {} is not supported: must be one of {}.'.format(
                    msg, op, valid) + 
                '\n' + LiftedRDDLSimulator._print_stack_trace(expr))
    
    @staticmethod
    def _check_arity(args, required, msg, expr):
        actual = len(args)
        if actual != required:
            raise RDDLInvalidNumberOfArgumentsError(
                '{} requires {} arguments, got {}.'.format(
                    msg, required, actual) + 
                '\n' + LiftedRDDLSimulator._print_stack_trace(expr))
    
    @staticmethod
    def _check_positive(value, strict, msg, expr):
        if strict:
            failed = not np.all(value > 0).item()
            message = '{} must be strictly positive, got {}.'
        else:
            failed = not np.all(value >= 0).item()
            message = '{} must be non-negative, got {}.'
        if failed:
            raise RDDLValueOutOfRangeError(
                message.format(msg, value) + 
                '\n' + LiftedRDDLSimulator._print_stack_trace(expr))
    
    @staticmethod
    def _check_bounds(lb, ub, msg, expr):
        if not np.all(lb <= ub).item():
            raise RDDLValueOutOfRangeError(
                'Bounds of {} are invalid:'.format(msg) + 
                'max value {} must be >= min value {}.'.format(ub, lb) + 
                '\n' + LiftedRDDLSimulator._print_stack_trace(expr))
            
    @staticmethod
    def _check_range(value, lb, ub, msg, expr):
        if not np.all((value >= lb) & (value <= ub)).item():
            raise RDDLValueOutOfRangeError(
                '{} must be in the range [{}, {}], got {}.'.format(
                    msg, lb, ub, value) + 
                '\n' + LiftedRDDLSimulator._print_stack_trace(expr))
    
    @staticmethod
    def _raise_unsupported(msg, expr):
        raise RDDLNotImplementedError(
            '{} is not supported in the current RDDL version.'.format(msg) + 
            '\n' + LiftedRDDLSimulator._print_stack_trace(expr))
    
    @staticmethod
    def _raise_no_args(op, msg, expr):
        raise RDDLInvalidNumberOfArgumentsError(
            '{} operator {} has no arguments.'.format(msg, op) + 
            '\n' + LiftedRDDLSimulator._print_stack_trace(expr))
    
    # ===========================================================================
    # main sampling routines
    # ===========================================================================
    
    def check_state_invariants(self) -> None:
        '''Throws an exception if the state invariants are not satisfied.'''
        for _, invariant in enumerate(self.domain.invariants):
            sample = self._sample(invariant, [], self.subs)
            LiftedRDDLSimulator._check_type(sample, bool, 'Invariant', invariant)
            if not sample.item():
                raise RDDLStateInvariantNotSatisfiedError(
                    'Invariant is not satisfied.' + 
                    '\n' + LiftedRDDLSimulator._print_stack_trace(invariant))
    
    def check_action_preconditions(self) -> None:
        '''Throws an exception if the action preconditions are not satisfied.'''
        for _, precond in enumerate(self.domain.preconds):
            sample = self._sample(precond, [], self.subs)       
            LiftedRDDLSimulator._check_type(sample, bool, 'Precondition', precond)     
            if not sample.item():
                raise RDDLActionPreconditionNotSatisfiedError(
                    'Precondition is not satisfied.' + 
                    '\n' + LiftedRDDLSimulator._print_stack_trace(precond))
    
    def check_terminal_states(self) -> bool:
        '''return True if a terminal state has been reached.'''
        for _, terminal in enumerate(self.domain.terminals):
            sample = self._sample(terminal, [], self.subs)
            LiftedRDDLSimulator._check_type(sample, bool, 'Termination', terminal)     
            if sample.item():
                return True
        return False
    
    def sample_reward(self) -> float:
        '''Samples the current reward given the current state and action.'''
        sample = self._sample(self.domain.reward, [], self.subs)
        return float(sample.item())
    
    def reset(self) -> Dict:
        '''Resets the state variables to their initial values.'''
        self.subs = self.init_values.copy()  
        obs = {state: self.subs[state] for state in self.next_states.values()}        
        done = self.check_terminal_states()        
        return obs, done
    
    def _vectorize_actions(self, actions: Dict) -> Dict:
        new_actions = {action: np.copy(values) 
                       for action, values in self.noop_actions.items()}
        
        for action, value in actions.items():
            
            # parse the specified action
            name, objects = action, ''
            if '(' in action or ')' in action:
                parsed_action = self.action_pattern.match(action)
                if not parsed_action:
                    raise RDDLInvalidActionError(
                        'Action fluent <{}> is not valid, '.format(action) + 
                        'must be either <name> or <name>(<type1>,<type2>...).')
                name, objects = parsed_action.groups()
                
            # check for valid action syntax <name>(<types...>)
            if name not in self.noop_actions:
                raise RDDLInvalidActionError(
                    'Action fluent <{}> is not valid, '.format(name) + 
                    'must be in {{{}}}.'.format(
                        ', '.join(self.noop_actions.keys())))
            
            # check that the value is compatible with RDDL definition
            prange_val = type(value)
            prange_req = new_actions[name].dtype
            if not np.can_cast(prange_val, prange_req, casting='safe'):
                raise RDDLTypeError(
                    'Value for action fluent <{}> of type {} '.format(
                        action, prange_val) + 
                    'cannot be cast to type {}.'.format(prange_req))
            
            # check that the number of arguments is correct
            objects = [obj.strip() for obj in objects.split(',')]
            len_val = len(objects)
            len_req = len(self.pvar_types[name])
            if len_val != len_req:
                raise RDDLInvalidActionError(
                    'Action fluent <{}> takes {} type arguments, '.format(
                        name, len_req) + 'given {}.'.format(objects))     
            
            # update the internal action arrays
            coords = self._objects_to_coordinates(
                objects, 'action-fluent <{}>'.format(action))
            new_actions[name][coords] = value
            
        return new_actions
    
    def step(self, actions: Dict) -> Dict:
        '''Samples and returns the next state from the CPF expressions.
        
        :param actions: a dict mapping current action fluent to their values
        '''
        actions = self._vectorize_actions(actions)        
        subs = self.subs
        subs.update(actions)
        
        for _, cpfs in self.levels.items():
            for cpf in cpfs:
                ptypes = self.cpf_types[cpf]
                expr = self.static.cpfs[cpf]
                subs[cpf] = self._sample(expr, ptypes, subs)
        reward = self.sample_reward()
        
        for next_state, state in self.next_states.items():
            subs[state] = subs[next_state]
        obs = {state: subs[state] for state in self.next_states.values()}
        
        done = self.check_terminal_states()
        
        return obs, reward, done
        
    # ===========================================================================
    # start of sampling subroutines
    # ===========================================================================
    
    def _sample(self, expr, objects, subs):
        etype, _ = expr.etype
        if etype == 'constant':
            return self._sample_constant(expr, objects, subs)
        elif etype == 'pvar':
            return self._sample_pvar(expr, objects, subs)
        elif etype == 'arithmetic':
            return self._sample_arithmetic(expr, objects, subs)
        elif etype == 'relational':
            return self._sample_relational(expr, objects, subs)
        elif etype == 'boolean':
            return self._sample_logical(expr, objects, subs)
        elif etype == 'aggregation':
            return self._sample_aggregation(expr, objects, subs)
        elif etype == 'func':
            return self._sample_func(expr, objects, subs)
        elif etype == 'control':
            return self._sample_control(expr, objects, subs)
        elif etype == 'randomvar':
            return self._sample_random(expr, objects, subs)
        else:
            raise RDDLNotImplementedError(
                'Internal error: expression {} is not recognized.'.format(expr))
                
    # ===========================================================================
    # leaves
    # ===========================================================================
    
    def _get_subs_mapping(self, objects_has, objects_req, expr):
        if hasattr(expr, 'cached_sub_map'):
            return expr.cached_sub_map
        
        # reached limit on number of valid dimensions (52)
        valid_symbols = LiftedRDDLSimulator.VALID_SYMBOLS
        n_valid = len(valid_symbols)
        n_req = len(objects_req)
        if n_req > n_valid:
            raise RDDLNotImplementedError(
                'Up to {}-D are supported, but variable <{}> is {}-D.'.format(
                    n_valid, expr.args[0], n_req) + 
                '\n' + LiftedRDDLSimulator._print_stack_trace(expr))
                
        # find a map permutation(a,b,c...) -> (a,b,c...) for the correct einsum
        lhs = [None] * len(objects_has)
        new_dims = []
        for i_req, (obj_req, type_req) in enumerate(objects_req):
            new_dim = True
            for i_has, (obj_has, type_has) in enumerate(objects_has):
                if obj_has == obj_req:
                    lhs[i_has] = valid_symbols[i_req]
                    new_dim = False
                    
                    # check evaluation matches the definition in pvariables {...}
                    if type_req != type_has:  
                        raise RDDLInvalidObjectError(
                            'Argument <{}> of variable <{}> '.format(
                                obj_req, expr.args[0]) + 
                            'expects type <{}>, got <{}>.'.format(
                                type_has, type_req) +
                            '\n' + LiftedRDDLSimulator._print_stack_trace(expr))
            
            # need to expand the shape of the value array
            if new_dim:
                lhs.append(valid_symbols[i_req])
                new_dims.append(len(self.objects[type_req]))
                
        # safeguard against any free types
        free_types = [objects_has[1][i] for i, p in enumerate(lhs) if p is None]
        if free_types:
            raise RDDLInvalidNumberOfArgumentsError(
                'Variable <{}> has free parameter(s) {}.'.format(
                    expr.args[0], free_types) + 
                '\n' + LiftedRDDLSimulator._print_stack_trace(expr))
            
        lhs = ''.join(lhs)
        rhs = valid_symbols[:n_req]
        permute = lhs + ' -> ' + rhs
        identity = lhs == rhs
        new_dims = tuple(new_dims)   
        expr.cached_sub_map = (permute, identity, new_dims)
        
        if self.debug:
            warnings.warn(
                'caching map info for einsum:' + 
                '\n\t' + 'inputs ={}'.format(objects_has) + 
                '\n\t' + 'targets={}'.format(objects_req) + 
                '\n\t' + 'einsum ={}'.format(permute) + 
                '\n\t' + 'expr   ={}'.format(expr) + '\n'
            )
            
        return expr.cached_sub_map
    
    def _sample_constant(self, expr, objects, _):
        permute, identity, new_dims = self._get_subs_mapping([], objects, expr)
        arg = np.asarray(expr.args)
        sample = arg
        if new_dims:
            new_axes = (1,) * len(new_dims)        
            sample = np.reshape(arg, newshape=arg.shape + new_axes) 
            sample = np.broadcast_to(sample, shape=arg.shape + new_dims)
        if not identity:
            sample = np.einsum(permute, sample)
        return sample
    
    def _sample_pvar(self, expr, objects, subs):
        _, name = expr.etype
        args = expr.args
        LiftedRDDLSimulator._check_arity(args, 2, 'pvar ' + name, expr)
        
        var, pvars = args        
        if var not in subs:
            raise RDDLUndefinedVariableError(
                'Variable <{}> is not defined in the instance.'.format(var) + 
                '\n' + LiftedRDDLSimulator._print_stack_trace(expr))
            
        arg = subs[var]
        if arg is None:
            raise RDDLUndefinedVariableError(
                'Variable <{}> is referenced before assignment.'.format(var) + 
                '\n' + LiftedRDDLSimulator._print_stack_trace(expr))
        
        if pvars is None:
            pvars = []
        types = self.pvar_types[var]
        len_has = len(pvars)
        len_req = len(types)
        if len_has != len_req:
            raise RDDLInvalidNumberOfArgumentsError(
                'Variable <{}> requires {} parameters, got {}.'.format(
                    var, len_req, len_has) + 
                '\n' + LiftedRDDLSimulator._print_stack_trace(expr))
        
        pvars = tuple(zip(pvars, types))
        permute, identity, new_dims = self._get_subs_mapping(pvars, objects, expr)
        arg = np.asarray(arg)
        sample = arg
        if new_dims:
            new_axes = (1,) * len(new_dims)
            sample = np.reshape(arg, newshape=arg.shape + new_axes) 
            sample = np.broadcast_to(sample, shape=arg.shape + new_dims)
        if not identity:
            sample = np.einsum(permute, sample)
        return sample
    
    # ===========================================================================
    # mathematical
    # ===========================================================================
    
    def _sample_arithmetic(self, expr, objects, subs):
        _, op = expr.etype
        valid_ops = self.ARITHMETIC_OPS
        LiftedRDDLSimulator._check_op(op, valid_ops, 'Arithmetic', expr)
        
        args = expr.args        
        n = len(args)        
        if n == 1 and op == '-':
            arg, = args
            return -1 * self._sample(arg, objects, subs)
        
        elif n == 2:
            lhs, rhs = args
            lhs = 1 * self._sample(lhs, objects, subs)
            rhs = 1 * self._sample(rhs, objects, subs)
            return valid_ops[op](lhs, rhs)
        
        LiftedRDDLSimulator._check_arity(args, 2, 'Arithmetic operator', expr)
    
    def _sample_relational(self, expr, objects, subs):
        _, op = expr.etype
        args = expr.args
        valid_ops = self.RELATIONAL_OPS
        LiftedRDDLSimulator._check_op(op, valid_ops, 'Relational', expr)
        LiftedRDDLSimulator._check_arity(args, 2, 'Relational operator ' + op, expr)
        
        lhs, rhs = args
        lhs = 1 * self._sample(lhs, objects, subs)
        rhs = 1 * self._sample(rhs, objects, subs)
        return valid_ops[op](lhs, rhs)
    
    def _sample_logical(self, expr, objects, subs):
        _, op = expr.etype
        args = expr.args
        valid_ops = self.LOGICAL_OPS
        LiftedRDDLSimulator._check_op(op, valid_ops, 'Logical', expr)
        
        n = len(args)
        if n == 1 and op == '~':
            arg, = args
            arg = self._sample(arg, objects, subs)
            LiftedRDDLSimulator._check_type(
                arg, bool, 'Argument of logical operator ' + op, expr)
            return np.logical_not(arg)
        
        elif n == 2:
            lhs, rhs = args
            lhs = self._sample(lhs, objects, subs)
            rhs = self._sample(rhs, objects, subs)
            LiftedRDDLSimulator._check_types(
                lhs, rhs, bool, 'Arguments of logical operator ' + op, expr)
            return valid_ops[op](lhs, rhs)
        
        LiftedRDDLSimulator._check_arity(args, 2, 'Logical operator', expr)
        
    def _sample_aggregation(self, expr, objects, subs):
        _, op = expr.etype
        args = expr.args
        valid_ops = self.AGGREGATION_OPS
        LiftedRDDLSimulator._check_op(op, valid_ops, 'Aggregation', expr)

        * pvars, arg = args
        new_objects = objects + [p[1] for p in pvars]
        axis = tuple(range(len(objects), len(new_objects)))
        arg = self._sample(arg, new_objects, subs)
        
        if op == 'forall' or op == 'exists':
            LiftedRDDLSimulator._check_type(
                arg, bool, 'Argument of aggregation ' + op, expr)
        else:
            arg = 1 * arg
        return valid_ops[op](arg, axis=axis)
    
    def _sample_func(self, expr, objects, subs):
        _, name = expr.etype
        args = expr.args
        if isinstance(args, Expression):
            args = (args,)
            
        if name in self.UNARY:
            LiftedRDDLSimulator._check_arity(
                args, 1, 'Unary function ' + name, expr)
            arg, = args
            arg = 1 * self._sample(arg, objects, subs)
            return self.UNARY[name](arg)
        
        elif name in self.BINARY:
            LiftedRDDLSimulator._check_arity(
                args, 2, 'Binary function ' + name, expr)
            lhs, rhs = args
            lhs = 1 * self._sample(lhs, objects, subs)
            rhs = 1 * self._sample(rhs, objects, subs)
            return self.BINARY[name](lhs, rhs)
        
        LiftedRDDLSimulator._raise_unsupported('Function ' + name, expr)
    
    # ===========================================================================
    # control flow
    # ===========================================================================
    
    def _sample_control(self, expr, objects, subs):
        _, op = expr.etype
        args = expr.args
        LiftedRDDLSimulator._check_op(op, {'if'}, 'Control', expr)
        LiftedRDDLSimulator._check_arity(args, 3, 'If then else', expr)
        
        pred, arg1, arg2 = args
        pred = self._sample(pred, objects, subs)
        LiftedRDDLSimulator._check_type(pred, bool, 'Predicate', expr)
        
        if pred.size == 1:  # can short-circuit
            arg = arg1 if pred.item() else arg2
            return self._sample(arg, objects, subs)
        else:
            arg1 = self._sample(arg1, objects, subs)
            arg2 = self._sample(arg2, objects, subs)
            return np.where(pred, arg1, arg2)
        
    # ===========================================================================
    # random variables
    # ===========================================================================
    
    def _sample_random(self, expr, objects, subs):
        _, name = expr.etype
        if name == 'KronDelta':
            return self._sample_kron_delta(expr, objects, subs)        
        elif name == 'DiracDelta':
            return self._sample_dirac_delta(expr, objects, subs)
        elif name == 'Uniform':
            return self._sample_uniform(expr, objects, subs)
        elif name == 'Bernoulli':
            return self._sample_bernoulli(expr, objects, subs)
        elif name == 'Normal':
            return self._sample_normal(expr, objects, subs)
        elif name == 'Poisson':
            return self._sample_poisson(expr, objects, subs)
        elif name == 'Exponential':
            return self._sample_exponential(expr, objects, subs)
        elif name == 'Weibull':
            return self._sample_weibull(expr, objects, subs)        
        elif name == 'Gamma':
            return self._sample_gamma(expr, objects, subs)
        elif name == 'Binomial':
            return self._sample_binomial(expr, objects, subs)
        elif name == 'NegativeBinomial':
            return self._sample_negative_binomial(expr, objects, subs)
        elif name == 'Beta':
            return self._sample_beta(expr, objects, subs)
        elif name == 'Geometric':
            return self._sample_geometric(expr, objects, subs)
        elif name == 'Pareto':
            return self._sample_pareto(expr, objects, subs)
        elif name == 'Student':
            return self._sample_student(expr, objects, subs)
        elif name == 'Gumbel':
            return self._sample_gumbel(expr, objects, subs)
        else:  # no support for enum
            LiftedRDDLSimulator._raise_unsupported('Distribution ' + name, expr)

    def _sample_kron_delta(self, expr, objects, subs):
        args = expr.args
        LiftedRDDLSimulator._check_arity(args, 1, 'KronDelta', expr)
        
        arg, = args
        arg = self._sample(arg, objects, subs)
        LiftedRDDLSimulator._check_type(
            arg, bool, 'Argument of KronDelta', expr)
        return arg
    
    def _sample_dirac_delta(self, expr, objects, subs):
        args = expr.args
        LiftedRDDLSimulator._check_arity(args, 1, 'DiracDelta', expr)
        
        arg, = args
        arg = self._sample(arg, objects, subs)
        LiftedRDDLSimulator._check_type(
            arg, LiftedRDDLSimulator.REAL, 'Argument of DiracDelta', expr)        
        return arg
    
    def _sample_uniform(self, expr, objects, subs):
        args = expr.args
        LiftedRDDLSimulator._check_arity(args, 2, 'Uniform', expr)

        lb, ub = args
        lb = self._sample(lb, objects, subs)
        ub = self._sample(ub, objects, subs)
        LiftedRDDLSimulator._check_bounds(lb, ub, 'Uniform', expr)
        return self.rng.uniform(lb, ub)      
    
    def _sample_bernoulli(self, expr, objects, subs):
        args = expr.args
        LiftedRDDLSimulator._check_arity(args, 1, 'Bernoulli', expr)
        
        pr, = args
        pr = self._sample(pr, objects, subs)
        LiftedRDDLSimulator._check_range(pr, 0, 1, 'Bernoulli p', expr)
        return self.rng.uniform(size=pr.shape) <= pr
    
    def _sample_normal(self, expr, objects, subs):
        args = expr.args
        LiftedRDDLSimulator._check_arity(args, 2, 'Normal', expr)
        
        mean, var = args
        mean = self._sample(mean, objects, subs)
        var = self._sample(var, objects, subs)
        LiftedRDDLSimulator._check_positive(var, False, 'Normal variance', expr)  
        std = np.sqrt(var)
        return self.rng.normal(mean, std)
    
    def _sample_poisson(self, expr, objects, subs):
        args = expr.args
        LiftedRDDLSimulator._check_arity(args, 1, 'Poisson', expr)
        
        rate, = args
        rate = self._sample(rate, objects, subs)
        LiftedRDDLSimulator._check_positive(rate, False, 'Poisson rate', expr)        
        return self.rng.poisson(rate)
    
    def _sample_exponential(self, expr, objects, subs):
        args = expr.args
        LiftedRDDLSimulator._check_arity(args, 1, 'Exponential', expr)
        
        scale, = expr.args
        scale = self._sample(scale, objects, subs)
        LiftedRDDLSimulator._check_positive(scale, True, 'Exponential rate', expr)
        return self.rng.exponential(scale)
    
    def _sample_weibull(self, expr, objects, subs):
        args = expr.args
        LiftedRDDLSimulator._check_arity(args, 2, 'Weibull', expr)
        
        shape, scale = args
        shape = self._sample(shape, objects, subs)
        scale = self._sample(scale, objects, subs)
        LiftedRDDLSimulator._check_positive(shape, True, 'Weibull shape', expr)
        LiftedRDDLSimulator._check_positive(scale, True, 'Weibull scale', expr)
        return scale * self.rng.weibull(shape)
    
    def _sample_gamma(self, expr, objects, subs):
        args = expr.args
        LiftedRDDLSimulator._check_arity(args, 2, 'Gamma', expr)
        
        shape, scale = args
        shape = self._sample(shape, objects, subs)
        scale = self._sample(scale, objects, subs)
        LiftedRDDLSimulator._check_positive(shape, True, 'Gamma shape', expr)            
        LiftedRDDLSimulator._check_positive(scale, True, 'Gamma scale', expr)        
        return self.rng.gamma(shape, scale)
    
    def _sample_binomial(self, expr, objects, subs):
        args = expr.args
        LiftedRDDLSimulator._check_arity(args, 2, 'Binomial', expr)
        
        count, pr = args
        count = self._sample(count, objects, subs)
        pr = self._sample(pr, objects, subs)
        LiftedRDDLSimulator._check_type(
            count, LiftedRDDLSimulator.INT, 'Binomial count', expr)
        LiftedRDDLSimulator._check_positive(count, False, 'Binomial count', expr)
        LiftedRDDLSimulator._check_range(pr, 0, 1, 'Binomial p', expr)
        return self.rng.binomial(count, pr)
    
    def _sample_negative_binomial(self, expr, objects, subs):
        args = expr.args
        LiftedRDDLSimulator._check_arity(args, 2, 'NegativeBinomial', expr)
        
        count, pr = args
        count = self._sample(count, objects, subs)
        pr = self._sample(pr, objects, subs)
        LiftedRDDLSimulator._check_positive(
            count, True, 'NegativeBinomial r', expr)
        LiftedRDDLSimulator._check_range(
            pr, 0, 1, 'NegativeBinomial p', expr)        
        return self.rng.negative_binomial(count, pr)
    
    def _sample_beta(self, expr, objects, subs):
        args = expr.args
        LiftedRDDLSimulator._check_arity(args, 2, 'Beta', expr)
        
        shape, rate = args
        shape = self._sample(shape, objects, subs)
        rate = self._sample(rate, objects, subs)
        LiftedRDDLSimulator._check_positive(shape, True, 'Beta shape', expr)
        LiftedRDDLSimulator._check_positive(rate, True, 'Beta rate', expr)        
        return self.rng.beta(shape, rate)

    def _sample_geometric(self, expr, objects, subs):
        args = expr.args
        LiftedRDDLSimulator._check_arity(args, 1, 'Geometric', expr)
        
        pr, = args
        pr = self._sample(pr, objects, subs)
        LiftedRDDLSimulator._check_range(pr, 0, 1, 'Geometric p', expr)        
        return self.rng.geometric(pr)
    
    def _sample_pareto(self, expr, objects, subs):
        args = expr.args
        LiftedRDDLSimulator._check_arity(args, 2, 'Pareto', expr)
        
        shape, scale = args
        shape = self._sample(shape, objects, subs)
        scale = self._sample(scale, objects, subs)
        LiftedRDDLSimulator._check_positive(shape, True, 'Pareto shape', expr)        
        LiftedRDDLSimulator._check_positive(scale, True, 'Pareto scale', expr)        
        return scale * self.rng.pareto(shape)
    
    def _sample_student(self, expr, objects, subs):
        args = expr.args
        LiftedRDDLSimulator._check_arity(args, 1, 'Student', expr)
        
        df, = args
        df = self._sample(df, objects, subs)
        LiftedRDDLSimulator._check_positive(df, True, 'Student df', expr)            
        return self.rng.standard_t(df)

    def _sample_gumbel(self, expr, objects, subs):
        args = expr.args
        LiftedRDDLSimulator._check_arity(args, 2, 'Gumbel', expr)
        
        mean, scale = args
        mean = self._sample(mean, objects, subs)
        scale = self._sample(scale, objects, subs)
        LiftedRDDLSimulator._check_positive(scale, True, 'Gumbel scale', expr)
        return self.rng.gumbel(mean, scale)
        

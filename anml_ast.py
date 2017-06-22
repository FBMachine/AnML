from anml_typesystem import *

class SymbolTable(object):
    def __init__(self):
        self.tables = [{}]

    def push_table(self):
        self.tables.append({})
        return self.tables[-1]

    def pop_table(self):
        return self.tables.pop()

    def get(self, ident):
        for table in reversed(self.tables):
            if ident in table:
                return table[ident]
        return None

    def insert(self, ident, node, offset=-1):
        self.tables[offset][ident] = node

# Scope simplifies pushing a new scope to symbol table
class Scope(object):
    def __init__(self, symbol_table):
        self.symbol_table = symbol_table

    def __enter__(self):
        return self.symbol_table.push_table()

    def __exit__(self, type, value, bt):
        self.symbol_table.pop_table()

class Function(object):
    def __init__(self, name, params, table):
        self.name = name
        self.params = params
        self.table = table
        self.val_type, self.ret_type = build_function_type(params)
        self.inferring_types = False

        for param in self.params:
            param.undefined = False

    # defer setting body to support recursive calls
    def set_body(self, body):
        # closure support: capture value of any bound variables
        self.body = [expr.capture_vals() for expr in body]

    def infer_types(self, args=[]):
        # recurrent function guard: return if we're already inferring types in body
        if self.inferring_types:
            return

        self.inferring_types = True

        if args:
            arg_count = len(args)
            param_count = len(self.params)
            if arg_count != param_count:
                raise Exception("error: Function '" + self.name + "' expecting " + str(param_count) + ' args, got ' + str(arg_count) + '.')

            for i, arg in enumerate(args):
                unify(self.params[i].val_type, arg.val_type)

        # unify return value first in case of recursive call
        unify(self.ret_type, self.body[-1].val_type)

        for expr in self.body:
            expr.infer_types()

        self.inferring_types = False

    def __str__(self):
        type_name_gen.reset()
        return "def " + self.name + ' : ' + self.val_type.get_name() + ' = <Function>'

    def eval(self):
        return self

    def arity(self):
        return len(self.params)

    def call(self, args):
        for i, arg in enumerate(args):
            # call-by-value / eager evaluation
            result = arg.eval()
            if type(result) == Function:
                self.params[i].bind_value(arg.capture_vals())
            else:
                self.params[i].bind_value(Value(result, arg.val_type))

        result = None
        for expr in self.body:
            result = expr.eval()

        return result

class FunctionCall(object):
    def __init__(self, var, args, func_type, ret_type):
        self.var = var
        self.args = args
        self.val_type = ret_type
        self.func_type = func_type

    def get_function(self):
        func = self.var.capture_vals() if type(self.var) == Variable else self.var
        prev_var = self.var
        while func != prev_var and type(func) == Variable:
            prev_var = func
            func = func.capture_vals()

        return func if type(func) == Function else None

    def infer_types(self):
        unify(self.var.val_type, self.func_type)
        func = self.get_function()

        if func != None:
            unify(self.val_type, func.ret_type)
            func.infer_types(self.args)

    def capture_vals(self):
        self.args = [expr.capture_vals() for expr in self.args]

        return self

    def eval(self):
        func = self.get_function()
        return func.call(self.args)

class Binding(object):
    def __init__(self, var, expr):
        self.var = var
        self.expr = expr
        self.val_type = UnitType
        self.cached_val = None

    def infer_types(self):
        self.expr.infer_types()
        self.var.bind_value(self.expr)
        self.var.infer_types()

    def capture_vals(self):
        # closure support: capture value of any bound variables
        self.expr = self.expr.capture_vals()
        return self

    def __str__(self):
        return self.var.name + ' : ' + self.var.val_type.get_name() + ' = ' + self.cached_val

    def eval(self):
        self.cached_val = str(self.var.eval())
        return self

class Value(object):
    def __init__(self, value, val_type):
        self.value = value
        self.val_type = val_type

    def infer_types(self):
        pass

    def capture_vals(self):
        return self

    def eval(self):
        return self.value

    def __str__(self):
        return str(self.value)

class Variable(object):
    def __init__(self, name):
        self.name = name
        self.value_node = None
        self.val_type = TypeVariable()
        # this is cleared when appearing in lhs of binding or as a parameter
        self.undefined = True

    def bind_value(self, value_node):
        self.value_node = value_node
        self.undefined = False

    def infer_types(self):
        if self.value_node:
            unify(self.val_type, self.value_node.val_type)
        elif self.undefined:
            raise Exception("error: Variable '" + self.name + "' undefined.")

    def capture_vals(self):
        if not self.value_node:
            return self

        return self.value_node

    def eval(self):
        if not self.value_node:
            raise Exception("error: Variable '" + self.name + "' unbound.")

        return self.value_node.eval()

class IfElse(object):
    def __init__(self, cond, true_body, false_body):
        self.cond = cond
        self.true_body = true_body
        self.false_body = false_body
        self.val_type = TypeVariable()

    def infer_types(self):
        self.cond.infer_types()
        unify(BoolType, self.cond.val_type)
        unify(self.val_type, self.true_body[-1].val_type)
        unify(self.true_body[-1].val_type, self.false_body[-1].val_type)

        for expr in self.true_body:
            expr.infer_types()

        for expr in self.false_body:
            expr.infer_types()

    def capture_vals(self):
        self.cond = self.cond.capture_vals()
        self.true_body = [body.capture_vals() for body in self.true_body]
        self.false_body = [body.capture_vals() for body in self.false_body]
        return self

    def eval(self):
        if self.cond.eval() == True:
            result = None
            for expr in self.true_body:
                result = expr.eval()
            return result

        result = None
        for expr in self.false_body:
            result = expr.eval()
        return result

class Negation(object):
    def __init__(self, operand):
        self.operand = operand
        self.val_type = TypeVariable()

    def infer_types(self):
        # TODO: negation should only unify with int or float
        unify(self.val_type, self.operand.val_type)

    def capture_vals(self):
        self.operand = self.operand.capture_vals()
        return self

    def eval(self):
        return -self.operand.eval()

class BinaryOp(object):
    def __init__(self, op, lhs, rhs):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

        if op in ['+', '-', '*', '/', '**']:
            self.val_type = TypeVariable()
            self.inherit_child_type = True
        elif op in ['and', 'or', '<', '>', '<=', '>=', '==', '!=' ]:
            self.val_type = BoolType
            self.inherit_child_type = False

        # type to propagate to children (can be different from above)
        if op in ['and', 'or']:
            self.prop_val_type = BoolType
        else:
            self.prop_val_type = None

    def infer_types(self):
        if self.prop_val_type != None:
            unify(self.prop_val_type, self.lhs.val_type)

        self.lhs.infer_types()
        self.rhs.infer_types()

        unify(self.lhs.val_type, self.rhs.val_type)

        if self.inherit_child_type:
            unify(self.val_type, self.lhs.val_type)

    def capture_vals(self):
        self.lhs = self.lhs.capture_vals()
        self.rhs = self.rhs.capture_vals()
        return self

    def eval(self):
        if self.op == '*':
            return self.lhs.eval() * self.rhs.eval()
        elif self.op == '/':
            return self.lhs.eval() / self.rhs.eval()
        elif self.op == '+':
            return self.lhs.eval() + self.rhs.eval()
        elif self.op == '-':
            return self.lhs.eval() - self.rhs.eval()
        elif self.op == '**':
            return self.lhs.eval() ** self.rhs.eval()
        elif self.op == '<':
            return self.lhs.eval() < self.rhs.eval()
        elif self.op == '>':
            return self.lhs.eval() > self.rhs.eval()
        elif self.op == '<=':
            return self.lhs.eval() <= self.rhs.eval()
        elif self.op == '>=':
            return self.lhs.eval() >= self.rhs.eval()
        elif self.op == '==':
            return self.lhs.eval() == self.rhs.eval()
        elif self.op == '!=':
            return self.lhs.eval() != self.rhs.eval()
        elif self.op == 'and':
            return self.lhs.eval() and self.rhs.eval()
        elif self.op == 'or':
            return self.lhs.eval() or self.rhs.eval()
        return None

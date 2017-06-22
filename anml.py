import tokenize
from tokenize import generate_tokens, tok_name
from io import BytesIO
from random import random as rand
import sys, traceback

class TypeNameGenerator:
    type_names = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    type_count = 0

    def reset(self):
        self.type_count = 0

    def new_type_name(self):
        type_name = self.type_names[self.type_count]
        self.type_count += 1
        self.type_count %= len(self.type_names)
        return type_name

type_name_gen = TypeNameGenerator()

def complex_types_match(lhs, rhs):
    if type(lhs) != list or type(rhs) != list:
        return lhs == rhs

    if len(lhs) != len(rhs):
        return False

    for i in xrange(len(lhs)):
        if type(lhs[i]) == str:
            if lhs[i] != rhs[i]:
                return False
            continue
        lhs_type = lhs[i].get_type()
        rhs_type = rhs[i].get_type()
        if lhs_type.concrete_type != rhs_type.concrete_type:
            return False
    return True

def unify_complex_types(lhs, rhs):
    if type(lhs) != list or type(rhs) != list:
        return

    if len(lhs) != len(rhs):
        raise Exception("error: Attempting to unify types of different length.")

    for i in xrange(len(lhs)):
        if type(lhs[i]) == str:
            continue
        unify(lhs[i], rhs[i])

# union-find for unifying types during type inference
def unify(lhs, rhs):
    # find root for each type variable
    lhs_type = lhs.get_type()
    rhs_type = rhs.get_type()

    if rhs_type == lhs_type:
        return

    unify_complex_types(lhs_type.concrete_type, rhs_type.concrete_type)

    # check for type mismatch
    both_typed = lhs_type.concrete_type != None and rhs_type.concrete_type != None
    mismatch = not complex_types_match(lhs_type.concrete_type, rhs_type.concrete_type)
    if both_typed and mismatch:
        mismatch_str = lhs_type.get_name() + " != " + rhs_type.get_name()
        raise Exception("Type Mismatch: " + mismatch_str)

    # if one variable has concrete type, set it as root
    if rhs_type.concrete_type != None:
        lhs_type.set_type(rhs_type)
    elif lhs_type.concrete_type != None:
        rhs_type.set_type(lhs_type)
    else:
        # TODO: use rank to choose order instead
        # flip coin to choose parent to create a more balanced tree
        if rand() >= 0.5:
            lhs_type.set_type(rhs_type)
        else:
            rhs_type.set_type(lhs_type)

class TypeVariable(object):
    def __init__(self, concrete_type=None):
        self.concrete_type = concrete_type
        self.parent = None
        self.type_name = ''

    def get_name(self):
        type_node = self.get_type()
        if type(type_node.concrete_type) == list:
            name = ''
            for t in type_node.concrete_type:
                if hasattr(t, '__name__'):
                    name += t.__name__
                elif hasattr(t, 'get_name'):
                    name += t.get_name()
                elif type(t) == str:
                    name += t
                else:
                    raise Exception('Internal Error')
            return name
        elif type_node.concrete_type != None:
            return type_node.concrete_type.__name__
        elif type_node.type_name != '':
            return type_node.type_name
        else:
            type_node.type_name = type_name_gen.new_type_name()
            return type_node.type_name

    def get_type(self):
        type_node = self

        # find root
        while type_node.parent != None:
            type_node = type_node.parent

        return type_node

    # set new type variable root
    def set_type(self, new_type):
        old_parent = self.parent
        self.parent = new_type

        # flatten to minimize depth of tree
        while old_parent != None:
            old_parent = old_parent.parent
            if old_parent != None:
                old_parent.parent = new_type

class Unit:
    pass

IntType = TypeVariable(int)
FloatType = TypeVariable(float)
StringType = TypeVariable(str)
BoolType = TypeVariable(bool)
UnitType = TypeVariable(Unit)

def build_function_type(params):
    ret_type = TypeVariable()
    type_obj = ['(']
    for param in params:
        if len(type_obj) > 1:
            type_obj.append(', ')
        type_obj.append(param.val_type)
    type_obj += [')', ' -> ']
    type_obj.append(ret_type)
    func_type = TypeVariable(type_obj)

    return func_type, ret_type

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
        # capture closure: unwrap any bound variables
        self.body = [expr.unwrap() for expr in body]

    def infer_types(self, args=[]):
        # recurrent function guard: we're already inferring types in body
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
        # TODO: These types won't be valid until type inference is split into
        #       a separate phase. They currently will appear to be independent.
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
                self.params[i].bind_value(arg.unwrap())
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
        func = self.var.unwrap() if type(self.var) == Variable else self.var
        prev_var = self.var
        while func != prev_var and type(func) == Variable:
            prev_var = func
            func = func.unwrap()

        return func if type(func) == Function else None

    def infer_types(self):
        unify(self.var.val_type, self.func_type)
        func = self.get_function()

        if func != None:
            unify(self.val_type, func.ret_type)
            func.infer_types(self.args)

    def unwrap(self):
        self.args = [expr.unwrap() for expr in self.args]

        return self

    def eval(self):
        func = self.get_function()
        return func.call(self.args)

class Binding(object):
    def __init__(self, var, expr):
        self.var = var
        self.expr = expr
        self.val_type = UnitType

    def infer_types(self):
        self.expr.infer_types()
        self.var.bind_value(self.expr)
        self.var.infer_types()

    def unwrap(self):
        # capture closure: unwrap any bound variables
        self.expr = self.expr.unwrap()
        return self

    def eval(self):
        result = str(self.var.eval())
        return self.var.name + ' : ' + self.var.val_type.get_name() + ' = ' + result

class Expression(object):
    pass

class Value(Expression):
    def __init__(self, value, val_type):
        self.value = value
        self.val_type = val_type

    def infer_types(self):
        pass

    def unwrap(self):
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

    def unwrap(self):
        if not self.value_node:
            return self

        return self.value_node

    def eval(self):
        if not self.value_node:
            raise Exception("error: Variable '" + self.name + "' unbound.")

        return self.value_node.eval()

class IfElse(Expression):
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

    def unwrap(self):
        self.cond = self.cond.unwrap()
        self.true_body = [body.unwrap() for body in self.true_body]
        self.false_body = [body.unwrap() for body in self.false_body]
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

class BinaryOp(Expression):
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

    def unwrap(self):
        self.lhs = self.lhs.unwrap()
        self.rhs = self.rhs.unwrap()
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

class Lexer(object):
    def __init__(self):
        self.reserved = ['def', 'end', 'if', 'else', 'elif', 'return']
        self.line_num = 0

    def feed_input(self, line):
        tokens = generate_tokens(BytesIO(line.encode('utf-8')).readline)

        self.tokens = tokens
        # current token being processed
        self.token = tokens.next()
        # next token for peek-ahead
        self.next_token = tokens.next() if self.peek_type() != tokenize.ENDMARKER else None

    def peek_type(self):
        return self.token[0]

    def peek_val(self):
        return self.token[1]

    def peek_type_name(self):
        return tok_name[self.token[0]]

    def next(self):
        if self.peek_type() == tokenize.ENDMARKER:
            return

        self.token = self.next_token
        self.next_token = self.tokens.next() if self.peek_type() != tokenize.ENDMARKER else None

        # special case: want '->' to be treated as a single op
        if self.peek_val() == '-' and self.next_token and self.next_token[1] == '>':
            self.token = (self.token[0], '->')
            self.next_token = self.tokens.next()

        if self.peek_type() in [tokenize.NL, tokenize.NEWLINE]:
            self.line_num += 1

    def consume_whitespace(self):
        while self.peek_type() in [tokenize.INDENT, tokenize.DEDENT, tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT]:
            self.next()

    def consume_ident(self):
        if self.peek_type() == tokenize.NAME:
            ident = self.peek_val()
            if ident in self.reserved:
                return None
            self.next()
            return ident
        return None

    def consume_number(self):
        if self.peek_type() == tokenize.NUMBER:
            value = self.peek_val()
            self.next()
            return value
        return None

    def consume_string(self):
        if self.peek_type() == tokenize.STRING:
            string = self.peek_val()
            self.next()
            return string
        return None

    def consume_op(self, ops):
        if self.peek_val() in ops:
            op = self.peek_val()
            self.next()
            return op
        return None

    def consume_expected(self, token):
        if token == self.peek_val():
            self.next()
            return True
        return False

class Parser(object):
    def __init__(self, lexer):
        self.lexer = lexer
        self.binding_table = SymbolTable()

    def parse_string(self):
        string = self.lexer.consume_string()
        if string != None:
            return Value(string, StringType)
        return None

    def parse_number(self):
        is_neg = self.lexer.consume_expected('-')
        val = self.lexer.consume_number()
        if val != None:
            neg = '-' if is_neg else ''
            if '.' in val:
                return Value(float(neg + val), FloatType)
            else:
                return Value(int(neg + val), IntType)
        return None

    def parse_bool(self):
        if self.lexer.consume_expected('true'):
            return Value(True, BoolType)
        elif self.lexer.consume_expected('false'):
            return Value(False, BoolType)

        return None

    def parse_ident(self):
        is_neg = self.lexer.consume_expected('-')

        name = self.lexer.consume_ident()
        if name == None:
            return None

        var = self.binding_table.get(name)
        if var == None:
            var = Variable(name)
            self.binding_table.insert(name, var)

        # is this a function call?
        args = self.parse_arg_list()
        if args:
            func_type, ret_type = build_function_type(args)

            return FunctionCall(var, args, func_type, ret_type)

        return var

    def parse_paren_expression(self):
        if self.lexer.consume_expected('('):
            expr = self.parse_expression()
            if not expr:
                raise Exception("error: Missing expected expression.")
            if not self.lexer.consume_expected(')'):
                raise Exception("error: Missing closing ')'.")
            return expr
        return None

    def parse_atom(self):
        return self.parse_number() or self.parse_bool() or self.parse_string() or self.parse_paren_expression() or self.parse_ident()

    def parse_binary_expression(self, precedence=0):
        all_ops = [["or"],["and"],["<",">","==","!=",">=","<="],["+", "-"],["*", "/"], ["**"]]
        ops = all_ops[precedence]
        top_precedence = len(all_ops)-1

        lhs = None
        if precedence < (len(all_ops)-1):
            lhs = self.parse_expression(precedence+1)
        else:
            lhs = self.parse_atom()

        if lhs:
            bin_op = self.lexer.consume_op(ops)
            while bin_op:
                # min(top_precedence, precedence+1) causes lower precedence to
                #  group from left to right, and exponentiation to group
                #  from right to left
                rhs = self.parse_expression(min(top_precedence, precedence+1))
                if not rhs:
                    raise Exception("error: Missing expected expression.")
                lhs = BinaryOp(bin_op, lhs, rhs)
                bin_op = self.lexer.consume_op(ops)
            return lhs
        return None

    def parse_expression(self, precedence=0):
        return self.parse_if_else() or self.parse_binary_expression(precedence)

    def parse_block(self, end_tokens, allow_return=False):
        is_block = False
        if self.lexer.consume_expected('->'):
            self.lexer.consume_whitespace()
            body = [self.parse_expression()]
            self.lexer.consume_whitespace()
        elif self.lexer.consume_expected(':'):
            is_block = True
            body = []
            while not any(self.lexer.peek_val() == token for token in end_tokens):
                self.lexer.consume_whitespace()
                last_expr = self.lexer.consume_expected('return')
                if last_expr and not allow_return:
                    raise Exception("error: Unexpected 'return' from non-function scope.")
                expr = self.parse_binding()
                if not expr:
                    if last_expr and allow_return:
                        expr = Value(None, UnitType)
                    else:
                        raise Exception("error: Missing expected expression.")
                body.append(expr)
                self.lexer.consume_whitespace()
                if last_expr:
                    break
        else:
            raise Exception("error: Missing expected start of block ('->' or ':').")

        return body, is_block

    def parse_if_else(self):
        if self.lexer.consume_expected('if'):
            cond = self.parse_expression()

            with Scope(self.binding_table) as table:
                true_body, is_block = self.parse_block(['end', 'else'])

                if self.lexer.consume_expected('else'):
                    false_body, is_block = self.parse_block(['end'])
                else:
                    false_body = [Value(None, UnitType)]

                if is_block and not self.lexer.consume_expected('end'):
                    raise Exception("error: Missing expected 'end'.")

            return IfElse(cond, true_body, false_body)

        return None

    def parse_param_list(self):
        if not self.lexer.consume_expected('('):
            return None

        params = []
        while True:
            param_name = self.lexer.consume_ident()
            if param_name:
                var = Variable(param_name)
                self.binding_table.insert(param_name, var)
                params.append(var)

            if not self.lexer.consume_expected(','):
                break

        if not self.lexer.consume_expected(')'):
            raise Exception("error: Missing expected ')'.")

        return params

    def parse_arg_list(self):
        if not self.lexer.consume_expected('('):
            return None

        args = []
        while True:
            arg = self.parse_expression()
            if arg:
                args.append(arg)

            if not self.lexer.consume_expected(','):
                break

        if not self.lexer.consume_expected(')'):
            raise Exception("error: Missing expected ')'.")

        return args

    def parse_def(self):
        if self.lexer.consume_expected('def'):
            ident = self.lexer.consume_ident()
            with Scope(self.binding_table) as table:
                params = self.parse_param_list()

                # insert symbol before parsing body to support recursive calls
                func = Function(ident, params, table)
                func_var = Variable(ident)
                func_var.bind_value(func)
                self.binding_table.insert(ident, func_var, -2)

                body, is_block = self.parse_block(['end'], True)

                # if is_block and not self.lexer.consume_expected('end'):
                #     raise Exception("Missing expected 'end'.")

                func.set_body(body)

            return func

        return None

    def parse_binding(self):
        lhs = self.parse_expression()

        if self.lexer.consume_expected('='):
            if type(lhs) != Variable:
                raise Exception("error: Expected Variable on binding left hand side, got: " + type(lhs).__name__)
            rhs = self.parse_expression()
            if rhs == None:
                raise Exception("error: Expected expression on right hand side, got: " + type(rhs).__name__)
            if not lhs.undefined:
                # shadow previous definition
                lhs = Variable(lhs.name)
            self.binding_table.insert(lhs.name, lhs)
            return Binding(lhs, rhs)

        return lhs

    def parse_top_level(self):
        self.lexer.consume_whitespace()
        return self.parse_def() or self.parse_binding()

def REPL():
    lexer = Lexer()
    parser = Parser(lexer)
    last_bt = None

    while True:
        next_line = raw_input('>>> ')
        line = ''
        if next_line.endswith(':') or next_line.endswith('->'):
            line = next_line
            while next_line != '':
                next_line = raw_input('... ')
                if next_line != '':
                    line += '\n' + next_line
        else:
            line = next_line

        if line == 'exit' or line == 'exit()':
            break
        if line == 'bt()' and last_bt:
            traceback.print_tb(last_bt)
            continue

        lexer.feed_input(line)

        done = False
        try:
            while not done:
                done = True
                expr = parser.parse_top_level()
                if expr:
                    expr.infer_types()
                    result = expr.eval()
                    if result != None:
                        print str(result)
                    done = False
                    continue
        except Exception as ex:
            print ex
            exc_type, exc_value, exc_traceback = sys.exc_info()
            last_bt = exc_traceback

def eval_file(filename):
    lexer = Lexer()
    parser = Parser(lexer)
    program = []
    error_count = 0

    with open(filename, 'r') as infile:
        print "Compiling " + filename + '...'
        lines = infile.read()
        lexer.feed_input(lines)

        done = False
        while not done:
            try:
                done = True
                expr = parser.parse_top_level()
                if expr:
                    expr.infer_types()
                    program.append(expr)
                    done = False
                    continue
            except Exception as ex:
                print filename + '(' + str(lexer.line_num) + '): ' + str(ex)
                error_count += 1
                done = False
                # exc_type, exc_value, exc_traceback = sys.exc_info()
                # traceback.print_tb(exc_traceback)

        if error_count > 0:
            print '\nCompilation failed with ' + str(error_count) + ' error(s).'
            exit(-1)

        for expr in program:
            result = expr.eval()
            if result != None:
                print str(result)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        REPL()
    else:
        eval_file(sys.argv[1])

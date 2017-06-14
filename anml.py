import tokenize
from tokenize import generate_tokens, tok_name
from io import BytesIO
from random import random as rand
import sys, traceback

# union-find for unifying types during type inference
def unify(lhs, rhs):
    # find root for each type variable
    lhs_type = lhs.get_type()
    rhs_type = rhs.get_type()

    if rhs_type == lhs_type:
        return

    # check for type mismatch
    both_typed = lhs_type.concrete_type != None and rhs_type.concrete_type != None
    mismatch = lhs_type.concrete_type != rhs_type.concrete_type
    if both_typed and mismatch:
        mismatch_str = lhs_type.concrete_type.__name__ + " != " + rhs_type.concrete_type.__name__
        raise Exception("Type Mismatch: " + mismatch_str)

    # if one variable has concrete type, set it as root
    if rhs_type.concrete_type != None:
        lhs.set_type(rhs_type)
    elif lhs_type.concrete_type != None:
        rhs.set_type(lhs_type)
    else:
        # TODO: use rank to choose order instead
        # flip coin to choose parent to create a more balanced tree
        if rand() >= 0.5:
            lhs.set_type(rhs_type)
        else:
            rhs.set_type(lhs_type)

class TypeVariable(object):
    def __init__(self, concrete_type=None):
        self.concrete_type = concrete_type
        self.parent = None

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
        self.ret_type = TypeVariable()

    # defer setting body to support recursive calls
    def set_body(self, body):
        # capture closure: unwrap any bound variables
        self.body = [expr.unwrap() for expr in body]
        unify(self.ret_type, self.body[-1].val_type)

    def run(self):
        return None

    def arity(self):
        return len(self.params)

    def call(self, args):
        for i, arg in enumerate(args):
            # call-by-value / eager evaluation
            self.params[i].bind_value(Value(arg.run(), arg.val_type))

        result = None
        for expr in self.body:
            result = expr.run()

        return result

class FunctionCall(object):
    def __init__(self, func, args):
        arg_count = len(args)
        param_count = len(func.params)
        if arg_count != param_count:
            raise Exception("Function '" + func.name + "' expecting " + str(param_count) + ' args, got ' + str(arg_count) + '.')
        self.func = func
        self.args = args
        self.val_type = self.func.ret_type

    def unwrap(self):
        self.args = [expr.unwrap() for expr in self.args]

        return self

    def run(self):
        return self.func.call(self.args)

class Binding(object):
    def __init__(self, var, expr):
        self.var = var
        self.expr = expr

    def unwrap(self):
        # capture closure: unwrap any bound variables
        self.expr = self.expr.unwrap()
        return self

    def run(self):
        self.var.bind_value(self.expr)
        return self.var.run()

class Expression(object):
    pass

class Value(Expression):
    def __init__(self, value, val_type):
        self.value = value
        self.val_type = val_type

    def run(self):
        return self.value

    def unwrap(self):
        return self

    def __str__(self):
        return str(self.value)

class Variable(object):
    def __init__(self, name):
        self.name = name
        self.val_type = TypeVariable()
        self.value_node = None

    def bind_value(self, value_node):
        self.value_node = value_node
        unify(self.val_type, value_node.val_type)

    def unwrap(self):
        if not self.value_node:
            return self

        return self.value_node

    def run(self):
        if not self.value_node:
            raise Exception("Variable '" + self.name + "' unbound.")

        return self.value_node.run()

class IfElse(Expression):
    def __init__(self, cond, true_body, false_body):
        self.cond = cond
        self.true_body = true_body
        self.false_body = false_body
        self.val_type = TypeVariable()

    def unwrap(self):
        self.cond = self.cond.unwrap()
        self.true_body = [body.unwrap() for body in self.true_body]
        self.false_body = [body.unwrap() for body in self.false_body]
        return self

    def run(self):
        unify(BoolType, self.cond.val_type)
        unify(self.val_type, self.true_body[-1].val_type)
        unify(self.true_body[-1].val_type, self.false_body[-1].val_type)

        if self.cond.run() == True:
            result = None
            for expr in self.true_body:
                result = expr.run()
            return result

        result = None
        for expr in self.false_body:
            result = expr.run()
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
            self.prop_val_type = TypeVariable()

    def unwrap(self):
        self.lhs = self.lhs.unwrap()
        self.rhs = self.rhs.unwrap()
        return self

    def run(self):
        if self.prop_val_type != None:
            unify(self.prop_val_type, self.lhs.val_type)

        if self.inherit_child_type:
            unify(self.val_type, self.lhs.val_type)

        unify(self.lhs.val_type, self.rhs.val_type)

        if self.op == '*':
            return self.lhs.run() * self.rhs.run()
        elif self.op == '/':
            return self.lhs.run() / self.rhs.run()
        elif self.op == '+':
            return self.lhs.run() + self.rhs.run()
        elif self.op == '-':
            return self.lhs.run() - self.rhs.run()
        elif self.op == '**':
            return self.lhs.run() ** self.rhs.run()
        elif self.op == '<':
            return self.lhs.run() < self.rhs.run()
        elif self.op == '>':
            return self.lhs.run() > self.rhs.run()
        elif self.op == '<=':
            return self.lhs.run() <= self.rhs.run()
        elif self.op == '>=':
            return self.lhs.run() >= self.rhs.run()
        elif self.op == '==':
            return self.lhs.run() == self.rhs.run()
        elif self.op == '!=':
            return self.lhs.run() != self.rhs.run()
        elif self.op == 'and':
            return self.lhs.run() and self.rhs.run()
        elif self.op == 'or':
            return self.lhs.run() or self.rhs.run()
        return None

class Lexer(object):
    def __init__(self):
        self.reserved = ['def', 'end', 'if', 'else', 'elif']

    def feed_line(self, line):
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

    def consume_whitespace(self):
        while self.peek_type() in [tokenize.INDENT, tokenize.DEDENT, tokenize.NL, tokenize.NEWLINE]:
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
            return FunctionCall(var, args)

        return var

    def parse_paren_expression(self):
        if self.lexer.consume_expected('('):
            expr = self.parse_expression()
            if not expr:
                raise Exception("Missing expected expression.")
            if not self.lexer.consume_expected(')'):
                raise Exception("Missing closing ')'.")
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
                    raise Exception("Missing expected expression.")
                lhs = BinaryOp(bin_op, lhs, rhs)
                bin_op = self.lexer.consume_op(ops)
            return lhs
        return None

    def parse_expression(self, precedence=0):
        return self.parse_if_else() or self.parse_binary_expression(precedence)

    def parse_block(self, end_tokens):
        is_block = False
        if self.lexer.consume_expected('->'):
            self.lexer.consume_whitespace()
            body = [self.parse_expression()]
            self.lexer.consume_whitespace()
        elif self.lexer.consume_expected(':'):
            is_block = True
            body = []
            while not any(self.lexer.peek_val(token) for token in end_tokens):
                self.lexer.consume_whitespace()
                expr = self.parse_binding()
                if not expr:
                    raise Exception("Missing expected expression.")
                body.append(expr)
                self.lexer.consume_whitespace()
        else:
            raise Exception("Missing expected start of block ('->' or ':').")

        return body, is_block

    def parse_if_else(self):
        if self.lexer.consume_expected('if'):
            cond = self.parse_expression()
            true_body, is_block = self.parse_block(['end', 'else'])

            if self.lexer.consume_expected('else'):
                false_body, is_block = self.parse_block(['end'])
            else:
                false_body = [Value(None, UnitType)]

            if is_block and not self.lexer.consume_expected('end'):
                raise Exception("Missing expected 'end'.")

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
            raise Exception("Missing expected ')'.")

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
            raise Exception("Missing expected ')'.")

        return args

    def parse_def(self):
        if self.lexer.consume_expected('def'):
            ident = self.lexer.consume_ident()
            with Scope(self.binding_table) as table:
                params = self.parse_param_list()

                # insert symbol before parsing body to support recursive calls
                func = Function(ident, params, table)
                self.binding_table.insert(ident, func, -2)

                body, is_block = self.parse_block(['end'])

                if is_block and not self.lexer.consume_expected('end'):
                    raise Exception("Missing expected 'end'.")

                func.set_body(body)

            return func

        return None

    def parse_binding(self):
        lhs = self.parse_expression()

        if self.lexer.consume_expected('='):
            if type(lhs) != Variable:
                raise Exception("Expected Variable on binding left hand side, got: " + type(lhs).__name__)
            self.binding_table.insert(lhs.name, lhs)
            rhs = self.parse_expression()
            if rhs == None:
                raise Exception("Expected expression on right hand side, got: " + type(rhs).__name__)
            return Binding(lhs, rhs)

        return lhs

    def parse_top_level(self):
        return self.parse_def() or self.parse_binding()

if __name__ == '__main__':
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

        lexer.feed_line(line)

        done = False
        try:
            while not done:
                done = True
                expr = parser.parse_top_level()
                if expr:
                    result = expr.run()
                    if result != None:
                        print str(result)
                    done = False
                    continue
        except Exception as ex:
            print ex
            exc_type, exc_value, exc_traceback = sys.exc_info()
            last_bt = exc_traceback

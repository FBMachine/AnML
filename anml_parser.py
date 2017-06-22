from anml_typesystem import *
from anml_ast import *

class Parser(object):
    def __init__(self, lexer):
        self.lexer = lexer
        self.binding_table = SymbolTable()

    def parse_string(self, is_neg):
        string = self.lexer.consume_string()
        if string != None:
            if is_neg:
                raise Exception("Unexpected negation of string.")
            return Value(string, StringType)
        return None

    def parse_number(self, is_neg):
        val = self.lexer.consume_number()
        if val != None:
            neg = '-' if is_neg else ''
            if '.' in val:
                return Value(float(neg + val), FloatType)
            else:
                return Value(int(neg + val), IntType)
        return None

    def parse_bool(self, is_neg):
        if self.lexer.consume_expected('true'):
            if is_neg:
                raise Exception("Unexpected negation of string.")
            return Value(True, BoolType)
        elif self.lexer.consume_expected('false'):
            if is_neg:
                raise Exception("Unexpected negation of string.")
            return Value(False, BoolType)

        return None

    def parse_ident(self, is_neg):
        name = self.lexer.consume_ident()
        if name == None:
            return None

        var = self.binding_table.get(name)
        if var == None:
            var = Variable(name)
            self.binding_table.insert(name, var)

        expr = var

        # is this a function call?
        args = self.parse_arg_list()
        if args:
            func_type, ret_type = build_function_type(args)

            expr = FunctionCall(var, args, func_type, ret_type)

        if is_neg:
            expr = Negation(expr)

        return expr

    def parse_paren_expression(self, is_neg):
        if self.lexer.consume_expected('('):
            expr = self.parse_expression()
            if not expr:
                raise Exception("error: Missing expected expression.")
            if not self.lexer.consume_expected(')'):
                raise Exception("error: Missing closing ')'.")

            if is_neg:
                expr = Negation(expr)
            return expr
        return None

    def parse_atom(self):
        is_neg = self.lexer.consume_expected('-')
        return self.parse_number(is_neg) or self.parse_bool(is_neg) or self.parse_string(is_neg) or self.parse_paren_expression(is_neg) or self.parse_ident(is_neg)

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

import tokenize
from tokenize import generate_tokens, tok_name
from io import BytesIO

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

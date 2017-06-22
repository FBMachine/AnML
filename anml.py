from anml_parser import Parser
from anml_lexer import Lexer

import sys, traceback

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

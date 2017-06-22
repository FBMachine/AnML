from random import random as rand

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

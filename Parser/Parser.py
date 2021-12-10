from position_finder import *

import string

DIGITS = '0123456789'
LETTERS = string.ascii_letters
LETTERS_DIGITS = LETTERS + DIGITS

class Error:
	def __init__(self, pos_start, pos_end, error_name, details):
		self.pos_start = pos_start
		self.pos_end = pos_end
		self.error_name = error_name
		self.details = details
	
	def as_string(self):
		result  = f'{self.error_name}: {self.details}\n'
		result += f'File {self.pos_start.fn}, line {self.pos_start.ln + 1}'
		result += '\n\n' + position_finder(self.pos_start.ftxt, self.pos_start, self.pos_end)
		return result

class IllegalCharError(Error):
	def __init__(self, pos_start, pos_end, details):
		super().__init__(pos_start, pos_end, 'Illegal Character', details)

class ExpectedCharError(Error):
	def __init__(self, pos_start, pos_end, details):
		super().__init__(pos_start, pos_end, 'Expected Character', details)

class InvalidSyntaxError(Error):
	def __init__(self, pos_start, pos_end, details=''):
		super().__init__(pos_start, pos_end, 'Invalid Syntax', details)

	def as_string(self):
		result  = self.generate_traceback()
		result += f'{self.error_name}: {self.details}'
		result += '\n\n' + position_finder(self.pos_start.ftxt, self.pos_start, self.pos_end)
		return result

	def generate_traceback(self):
		result = ''
		pos = self.pos_start
		ctx = self.context

		while ctx:
			result = f'  File {pos.fn}, line {str(pos.ln + 1)}, in {ctx.display_name}\n' + result
			pos = ctx.parent_entry_pos
			ctx = ctx.parent

		return 'Traceback (most recent call last):\n' + result

#######################################
# POSITION
#######################################

class Position:
	def __init__(self, idx, ln, col, fn, ftxt):
		self.idx = idx
		self.ln = ln
		self.col = col
		self.fn = fn
		self.ftxt = ftxt

	def advance(self, current_char=None):
		self.idx += 1
		self.col += 1

		if current_char == '\n':
			self.ln += 1
			self.col = 0

		return self

	def copy(self):
		return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)
INT				= 'INT'
FLOAT    	= 'FLOAT'
IDENTIFIER	= 'IDENTIFIER'
KEYWORD		= 'KEYWORD'
PLUS     	= 'PLUS'
MINUS    	= 'MINUS'
MUL      	= 'MUL'
DIV      	= 'DIV'
POWER				= 'POWER'
EQUAL					= 'EQ'
OPEN_PAREN   	= 'OPEN_PAREN'
CLOSE_PAREN   	= 'CLOSE_PAREN'
OPEN_BRACKET  = 'OPEN_BRACKET'
CLOSE_BRACKET = 'CLOSE_BRACKET'
EE					= 'EE'
NE					= 'NE'
LT					= 'LT'
GT					= 'GT'
LTE				= 'LTE'
GTE				= 'GTE'
COMMA			= 'COMMA'
EOF				= 'EOF'

KEYWORDS = [
	'class',
	'int',
	'void'
	'and',
	'or',
	'not',
	'if',
	'elif',
	'else',
	'for',
	'switch',
	'continue',
	'break',
	'while',
	'return',
]

class Token:
	def __init__(self, type_, value=None, pos_start=None, pos_end=None):
		self.type = type_
		self.value = value

		if pos_start:
			self.pos_start = pos_start.copy()
			self.pos_end = pos_start.copy()
			self.pos_end.advance()

		if pos_end:
			self.pos_end = pos_end.copy()

	def matches(self, type_, value):
		return self.type == type_ and self.value == value
	
	def __repr__(self):
		if self.value: return f'{self.type}:{self.value}'
		return f'{self.type}'

class Lexer:
	def __init__(self, fn, text):
		self.fn = fn
		self.text = text
		self.pos = Position(-1, 0, -1, fn, text)
		self.current_char = None
		self.advance()
	
	def advance(self):
		self.pos.advance(self.current_char)
		self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

	def make_tokens(self):
		tokens = []

		while self.current_char != None:
			if self.current_char in ' \t':
				self.advance()
			elif self.current_char in DIGITS:
				tokens.append(self.make_number())
			elif self.current_char in LETTERS:
				tokens.append(self.make_identifier())
			elif self.current_char == '+':
				tokens.append(Token(PLUS, pos_start=self.pos))
				self.advance()
			elif self.current_char == '-':
				tokens.append(self.make_minus_or_arrow())
			elif self.current_char == '*':
				tokens.append(Token(MUL, pos_start=self.pos))
				self.advance()
			elif self.current_char == '/':
				tokens.append(Token(DIV, pos_start=self.pos))
				self.advance()
			elif self.current_char == '^':
				tokens.append(Token(POWER, pos_start=self.pos))
				self.advance()
			elif self.current_char == '(':
				tokens.append(Token(OPEN_PAREN, pos_start=self.pos))
				self.advance()
			elif self.current_char == ')':
				tokens.append(Token(CLOSE_PAREN, pos_start=self.pos))
				self.advance()
			elif self.current_char == '!':
				token, error = self.make_not_equals()
				if error: return [], error
				tokens.append(token)
			elif self.current_char == '=':
				tokens.append(self.make_equals())
			elif self.current_char == '<':
				tokens.append(self.make_less_than())
			elif self.current_char == '>':
				tokens.append(self.make_greater_than())
			elif self.current_char == ',':
				tokens.append(Token(COMMA, pos_start=self.pos))
				self.advance()
			elif self.current_char == '{':
				tokens.append(Token(OPEN_BRACKET, pos_start=self.pos))
				self.advance()
			elif self.current_char == '}':
				tokens.append(Token(CLOSE_BRACKET, pos_start=self.pos))
				self.advance()
			else:
				pos_start = self.pos.copy()
				char = self.current_char
				self.advance()
				return [], IllegalCharError(pos_start, self.pos, "'" + char + "'")

		tokens.append(Token(EOF, pos_start=self.pos))
		return tokens, None

	def make_number(self):
		num_str = ''
		dot_count = 0
		pos_start = self.pos.copy()

		while self.current_char != None and self.current_char in DIGITS + '.':
			if self.current_char == '.':
				if dot_count == 1: break
				dot_count += 1
			num_str += self.current_char
			self.advance()

		if dot_count == 0:
			return Token(INT, int(num_str), pos_start, self.pos)
		else:
			return Token(FLOAT, float(num_str), pos_start, self.pos)

	def make_identifier(self):
		id_str = ''
		pos_start = self.pos.copy()

		while self.current_char != None and self.current_char in LETTERS_DIGITS + '_':
			id_str += self.current_char
			self.advance()

		tok_type = KEYWORD if id_str in KEYWORDS else IDENTIFIER
		return Token(tok_type, id_str, pos_start, self.pos)

	

	def make_not_equals(self):
		pos_start = self.pos.copy()
		self.advance()

		if self.current_char == '=':
			self.advance()
			return Token(NE, pos_start=pos_start, pos_end=self.pos), None

		self.advance()
		return None, ExpectedCharError(pos_start, self.pos, "'=' (after '!')")
	
	def make_equals(self):
		tok_type = EQUAL
		pos_start = self.pos.copy()
		self.advance()

		if self.current_char == '=':
			self.advance()
			tok_type = EE

		return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

	def make_less_than(self):
		tok_type = LT
		pos_start = self.pos.copy()
		self.advance()

		if self.current_char == '=':
			self.advance()
			tok_type = LTE

		return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

	def make_greater_than(self):
		tok_type = GT
		pos_start = self.pos.copy()
		self.advance()

		if self.current_char == '=':
			self.advance()
			tok_type = GTE

		return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

class NumberNode:
	def __init__(self, tok):
		self.tok = tok

		self.pos_start = self.tok.pos_start
		self.pos_end = self.tok.pos_end

	def __repr__(self):
		return f'{self.tok}'

class VarAccessNode:
	def __init__(self, var_name_tok):
		self.var_name_tok = var_name_tok

		self.pos_start = self.var_name_tok.pos_start
		self.pos_end = self.var_name_tok.pos_end

class VarAssignNode:
	def __init__(self, var_name_tok, value_node):
		self.var_name_tok = var_name_tok
		self.value_node = value_node

		self.pos_start = self.var_name_tok.pos_start
		self.pos_end = self.value_node.pos_end

class BinOpNode:
	def __init__(self, left_node, op_tok, right_node):
		self.left_node = left_node
		self.op_tok = op_tok
		self.right_node = right_node

		self.pos_start = self.left_node.pos_start
		self.pos_end = self.right_node.pos_end

	def __repr__(self):
		return f'({self.left_node}, {self.op_tok}, {self.right_node})'

class UnaryOpNode:
	def __init__(self, op_tok, node):
		self.op_tok = op_tok
		self.node = node

		self.pos_start = self.op_tok.pos_start
		self.pos_end = node.pos_end

	def __repr__(self):
		return f'({self.op_tok}, {self.node})'

class IfNode:
	def __init__(self, cases, else_case):
		self.cases = cases
		self.else_case = else_case

		self.pos_start = self.cases[0][0].pos_start
		self.pos_end = (self.else_case or self.cases[len(self.cases) - 1][0]).pos_end

class ForNode:
	def __init__(self, var_name_tok, start_value_node, end_value_node, step_value_node, body_node):
		self.var_name_tok = var_name_tok
		self.start_value_node = start_value_node
		self.end_value_node = end_value_node
		self.step_value_node = step_value_node
		self.body_node = body_node

		self.pos_start = self.var_name_tok.pos_start
		self.pos_end = self.body_node.pos_end

class WhileNode:
	def __init__(self, condition_node, body_node):
		self.condition_node = condition_node
		self.body_node = body_node

		self.pos_start = self.condition_node.pos_start
		self.pos_end = self.body_node.pos_end

class FuncDefNode:
	def __init__(self, var_name_tok, arg_name_toks, body_node):
		self.var_name_tok = var_name_tok
		self.arg_name_toks = arg_name_toks
		self.body_node = body_node

		if self.var_name_tok:
			self.pos_start = self.var_name_tok.pos_start
		elif len(self.arg_name_toks) > 0:
			self.pos_start = self.arg_name_toks[0].pos_start
		else:
			self.pos_start = self.body_node.pos_start

		self.pos_end = self.body_node.pos_end

class CallNode:
	def __init__(self, node_to_call, arg_nodes):
		self.node_to_call = node_to_call
		self.arg_nodes = arg_nodes

		self.pos_start = self.node_to_call.pos_start

		if len(self.arg_nodes) > 0:
			self.pos_end = self.arg_nodes[len(self.arg_nodes) - 1].pos_end
		else:
			self.pos_end = self.node_to_call.pos_end

#######################################
# PARSE RESULT
#######################################

class ParseResult:
	def __init__(self):
		self.error = None
		self.node = None
		self.last_registered_advance_count = 0
		self.advance_count = 0

	def register_advancement(self):
		self.last_registered_advance_count = 1
		self.advance_count += 1

	def register(self, res):
		self.last_registered_advance_count = res.advance_count
		self.advance_count += res.advance_count
		if res.error: self.error = res.error
		return res.node

	def success(self, node):
		self.node = node
		return self

	def failure(self, error):
		if not self.error or self.last_registered_advance_count == 0:
			self.error = error
		return self

#######################################
# PARSER
#######################################

class Parser:
	def __init__(self, tokens):
		self.tokens = tokens
		self.tok_idx = -1
		self.advance()

	def advance(self, ):
		self.tok_idx += 1
		if self.tok_idx < len(self.tokens):
			self.current_tok = self.tokens[self.tok_idx]
		return self.current_tok

	def parse(self):
		res = self.expr()
		if not res.error and self.current_tok.type != EOF:
			return res.failure(InvalidSyntaxError(
				self.current_tok.pos_start, self.current_tok.pos_end,
				"Expected '+', '-', '*', '/', '^', '==', '!=', '<', '>', <=', '>=', 'AND' or 'OR'"
			))
		return res

	###################################

	def expr(self):
		res = ParseResult()

		if self.current_tok.matches(KEYWORD, 'int'):
			res.register_advancement()
			self.advance()
		elif self.current_tok.matches(KEYWORD, 'void'):
			res.register_advancement()
			self.advance()

			if self.current_tok.type != IDENTIFIER:
				return res.failure(InvalidSyntaxError(
					self.current_tok.pos_start, self.current_tok.pos_end,
					"Expected identifier"
				))

			var_name = self.current_tok
			res.register_advancement()
			self.advance()

			if self.current_tok.type != EQUAL:
				return res.failure(InvalidSyntaxError(
					self.current_tok.pos_start, self.current_tok.pos_end,
					"Expected '='"
				))

			res.register_advancement()
			self.advance()
			expr = res.register(self.expr())
			if res.error: return res
			return res.success(VarAssignNode(var_name, expr))

		node = res.register(self.bin_op(self.comp_expr, ((KEYWORD, 'AND'), (KEYWORD, 'OR'))))

		if res.error:
			return res.failure(InvalidSyntaxError(
				self.current_tok.pos_start, self.current_tok.pos_end,
				"Expected 'INT', 'VOID' 'IF', 'FOR', 'WHILE, int, float, identifier, '+', '-', '(' or 'NOT'"
			))

		return res.success(node)

	def comp_expr(self):
		res = ParseResult()

		if self.current_tok.matches(KEYWORD, 'NOT'):
			op_tok = self.current_tok
			res.register_advancement()
			self.advance()

			node = res.register(self.comp_expr())
			if res.error: return res
			return res.success(UnaryOpNode(op_tok, node))
		
		node = res.register(self.bin_op(self.arith_expr, (EE, NE, LT, GT, LTE, GTE)))
		
		if res.error:
			return res.failure(InvalidSyntaxError(
				self.current_tok.pos_start, self.current_tok.pos_end,
				"Expected int, float, identifier, '+', '-', '(' or 'NOT'"
			))

		return res.success(node)

	def arith_expr(self):
		return self.bin_op(self.term, (PLUS, MINUS))

	def term(self):
		return self.bin_op(self.factor, (MUL, DIV))

	def factor(self):
		res = ParseResult()
		tok = self.current_tok

		if tok.type in (PLUS, MINUS):
			res.register_advancement()
			self.advance()
			factor = res.register(self.factor())
			if res.error: return res
			return res.success(UnaryOpNode(tok, factor))

		return self.power()

	def power(self):
		return self.bin_op(self.call, (POWER, ), self.factor)

	def call(self):
		res = ParseResult()
		atom = res.register(self.atom())
		if res.error: return res

		if self.current_tok.type == OPEN_PAREN:
			res.register_advancement()
			self.advance()
			arg_nodes = []

			if self.current_tok.type == CLOSE_PAREN:
				res.register_advancement()
				self.advance()
			else:
				arg_nodes.append(res.register(self.expr()))
				if res.error:
					return res.failure(InvalidSyntaxError(
						self.current_tok.pos_start, self.current_tok.pos_end,
						"Expected ')', 'VAR', 'IF', 'FOR', 'WHILE', 'FUN', int, float, identifier, '+', '-', '(' or 'NOT'"
					))

				while self.current_tok.type == COMMA:
					res.register_advancement()
					self.advance()

					arg_nodes.append(res.register(self.expr()))
					if res.error: return res

				if self.current_tok.type != CLOSE_PAREN:
					return res.failure(InvalidSyntaxError(
						self.current_tok.pos_start, self.current_tok.pos_end,
						f"Expected ',' or ')'"
					))

				res.register_advancement()
				self.advance()
			return res.success(CallNode(atom, arg_nodes))
		return res.success(atom)

	def atom(self):
		res = ParseResult()
		tok = self.current_tok

		if tok.type in (INT, FLOAT):
			res.register_advancement()
			self.advance()
			return res.success(NumberNode(tok))

		elif tok.type == IDENTIFIER:
			res.register_advancement()
			self.advance()
			return res.success(VarAccessNode(tok))

		elif tok.type == OPEN_PAREN:
			res.register_advancement()
			self.advance()
			expr = res.register(self.expr())
			if res.error: return res
			if self.current_tok.type == CLOSE_PAREN:
				res.register_advancement()
				self.advance()
				return res.success(expr)
			else:
				return res.failure(InvalidSyntaxError(
					self.current_tok.pos_start, self.current_tok.pos_end,
					"Expected ')'"
				))
		elif tok.matches(KEYWORD, 'class'):
			func_def = res.register(self.func_def())
			if res.error: return res
			return res.success(func_def)
		elif tok.matches(KEYWORD, 'if'):
			if_expr = res.register(self.if_expr())
			if res.error: return res
			return res.success(if_expr)

		elif tok.matches(KEYWORD, 'for'):
			for_expr = res.register(self.for_expr())
			if res.error: return res
			return res.success(for_expr)

		elif tok.matches(KEYWORD, 'while'):
			while_expr = res.register(self.while_expr())
			if res.error: return res
			return res.success(while_expr)

		elif tok.matches(KEYWORD, 'int'):
			func_def = res.register(self.func_def())
			if res.error: return res
			return res.success(func_def)

		elif tok.matches(KEYWORD, 'void'):
			func_def = res.register(self.func_def())
			if res.error: return res
			return res.success(func_def)
		return res.failure(InvalidSyntaxError(
			tok.pos_start, tok.pos_end,
			"Expected int, float, identifier, '+', '-', '(', 'IF', 'FOR', 'WHILE', 'FUN'"
		))

	def if_expr(self):
		res = ParseResult()
		cases = []
		else_case = None

		if not self.current_tok.matches(KEYWORD, 'if'):
			return res.failure(InvalidSyntaxError(
				self.current_tok.pos_start, self.current_tok.pos_end,
				f"Expected 'if'"
			))
		if not self.current_tok.matches(KEYWORD, 'OPEN_PAREN'):
			return res.failure(InvalidSyntaxError(
				self.current_tok.pos_start, self.current_tok.pos_end,
				f"Expected '('"
			))

		res.register_advancement()
		self.advance()

		condition = res.register(self.expr())
		if res.error: return res

		if not self.current_tok.matches(KEYWORD, 'CLOSE_PAREN'):
			return res.failure(InvalidSyntaxError(
				self.current_tok.pos_start, self.current_tok.pos_end,
				f"Expected ')'"
			))
		res.register_advancement()
		self.advance()

		if not self.current_tok.matches(KEYWORD, 'OPEN_BRACKET'):
				return res.failure(InvalidSyntaxError(
					self.current_tok.pos_start, self.current_tok.pos_end,
					f"Expected open bracket"
				))

		res.register_advancement()
		self.advance()

		expr = res.register(self.expr())
		if res.error: return res
		cases.append((condition, expr))
		if not self.current_tok.matches(KEYWORD, 'CLOSE_BRACKET'):
				return res.failure(InvalidSyntaxError(
					self.current_tok.pos_start, self.current_tok.pos_end,
					f"Expected closing bracket"
				))
		res.register_advancement()
		self.advance()
		
		
		while self.current_tok.matches(KEYWORD, 'ELIF'):
			res.register_advancement()
			self.advance()

			condition = res.register(self.expr())
			if res.error: return res

			if not self.current_tok.matches(KEYWORD, 'OPEN_BRACKET'):
				return res.failure(InvalidSyntaxError(
					self.current_tok.pos_start, self.current_tok.pos_end,
					f"Expected open bracket"
				))

			res.register_advancement()
			self.advance()

			expr = res.register(self.expr())
			if res.error: return res
			cases.append((condition, expr))
			

		if self.current_tok.matches(KEYWORD, 'ELSE'):
			res.register_advancement()
			self.advance()

			else_case = res.register(self.expr())
			if res.error: return res

		return res.success(IfNode(cases, else_case))

	def for_expr(self):
		res = ParseResult()

		if not self.current_tok.matches(KEYWORD, 'for'):
			return res.failure(InvalidSyntaxError(
				self.current_tok.pos_start, self.current_tok.pos_end,
				f"Expected 'for'"
			))

		res.register_advancement()
		self.advance()

		if self.current_tok.type != IDENTIFIER:
			return res.failure(InvalidSyntaxError(
				self.current_tok.pos_start, self.current_tok.pos_end,
				f"Expected identifier"
			))

		var_name = self.current_tok
		res.register_advancement()
		self.advance()

		if self.current_tok.type != EQUAL:
			return res.failure(InvalidSyntaxError(
				self.current_tok.pos_start, self.current_tok.pos_end,
				f"Expected '='"
			))
		
		res.register_advancement()
		self.advance()

		start_value = res.register(self.expr())
		if res.error: return res

		body = res.register(self.expr())
		if res.error: return res

		return res.success(ForNode(var_name, start_value,body))

	def while_expr(self):
		res = ParseResult()

		if not self.current_tok.matches(KEYWORD, 'while'):
			return res.failure(InvalidSyntaxError(
				self.current_tok.pos_start, self.current_tok.pos_end,
				f"Expected 'while'"
			))

		res.register_advancement()
		self.advance()

		condition = res.register(self.expr())
		if res.error: return res

		

		res.register_advancement()
		self.advance()

		body = res.register(self.expr())
		if res.error: return res

		return res.success(WhileNode(condition, body))

	def func_def(self):
		res = ParseResult()

		if not self.current_tok.matches(KEYWORD, 'int'):
			return res.failure(InvalidSyntaxError(
				self.current_tok.pos_start, self.current_tok.pos_end,
				f"Expected 'int' or 'void'"
			))

		res.register_advancement()
		self.advance()
		if not self.current_tok.matches(KEYWORD, 'void'):
			return res.failure(InvalidSyntaxError(
				self.current_tok.pos_start, self.current_tok.pos_end,
				f"Expected 'int' or 'void'"))
		if self.current_tok.type == IDENTIFIER:
			var_name_tok = self.current_tok
			res.register_advancement()
			self.advance()
			if self.current_tok.type != OPEN_PAREN:
				return res.failure(InvalidSyntaxError(
					self.current_tok.pos_start, self.current_tok.pos_end,
					f"Expected '('"
				))
		else:
			var_name_tok = None
			if self.current_tok.type != OPEN_PAREN:
				return res.failure(InvalidSyntaxError(
					self.current_tok.pos_start, self.current_tok.pos_end,
					f"Expected identifier or ')'"
				))
		
		res.register_advancement()
		self.advance()
		arg_name_toks = []

		if self.current_tok.type == IDENTIFIER:
			arg_name_toks.append(self.current_tok)
			res.register_advancement()
			self.advance()
			
			while self.current_tok.type == COMMA:
				res.register_advancement()
				self.advance()

				if self.current_tok.type != IDENTIFIER:
					return res.failure(InvalidSyntaxError(
						self.current_tok.pos_start, self.current_tok.pos_end,
						f"Expected identifier"
					))

				arg_name_toks.append(self.current_tok)
				res.register_advancement()
				self.advance()
			
			if self.current_tok.type != CLOSE_PAREN:
				return res.failure(InvalidSyntaxError(
					self.current_tok.pos_start, self.current_tok.pos_end,
					f"Expected ',' or ')'"
				))
		else:
			if self.current_tok.type != CLOSE_PAREN:
				return res.failure(InvalidSyntaxError(
					self.current_tok.pos_start, self.current_tok.pos_end,
					f"Expected identifier or ')'"
				))

		res.register_advancement()
		self.advance()
		node_to_return = res.register(self.expr())
		if res.error: return res

		return res.success(FuncDefNode(
			var_name_tok,
			arg_name_toks,
			node_to_return
		))

	###################################

	def bin_op(self, func_a, ops, func_b=None):
		if func_b == None:
			func_b = func_a
		
		res = ParseResult()
		left = res.register(func_a())
		if res.error: return res

		while self.current_tok.type in ops or (self.current_tok.type, self.current_tok.value) in ops:
			op_tok = self.current_tok
			res.register_advancement()
			self.advance()
			right = res.register(func_b())
			if res.error: return res
			left = BinOpNode(left, op_tok, right)

		return res.success(left)

class Context:
	def __init__(self, display_name, parent=None, parent_entry_pos=None):
		self.display_name = display_name
		self.parent = parent
		self.parent_entry_pos = parent_entry_pos
		self.symbol_table = None

def run(fn, text):
	lexer = Lexer(fn, text)
	tokens, error = lexer.make_tokens()
	if error: return None, error
	

	parser = Parser(tokens)
	ast = parser.parse()
	if ast.error: return None, ast.error


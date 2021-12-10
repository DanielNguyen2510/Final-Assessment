import Parser
f = open('example.txt','r')
while True:
	text = f.read()
	result, error = Parser.run('<stdin>', text)

	if error: print(error.as_string())
	elif result: print(result)

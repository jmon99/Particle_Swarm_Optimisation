import importlib.machinery
import os
import sys
from inspect import getmembers, isfunction

def line():
  print(u'\u2500' * 50)

file = sys.argv[1]
path = "./" + file
name = os.path.splitext(file)[0]

while True:
  print(os.getcwd())
  print(file, name, path)
  try:
    module = importlib.machinery.SourceFileLoader(name, path).load_module()
    break

  except Exception as e:
    print(e)
    print("file {} not found.".format(file))
    file = input("Please re-enter file or type EXIT!: ")
    if file == "EXIT!":
      break

functions = getmembers(module, isfunction)
total_tests = 0
passed = 0

for name, test in functions:

  if name[0:4] == "test":
    line()
    total_tests += 1
    result = test()
    if result == True:
      print(name + " Passed!")
      passed += 1
    else: 
      print(name)
      print(result)

    line()


line()
print("Passed {} out of {} tests!".format(passed, total_tests))
line()



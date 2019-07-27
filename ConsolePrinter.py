# -*- encoding: utf-8 -*-

from enum import Enum


class Printer:
	class color(Enum):
		black = [30, 40]
		red = [31, 41]
		green = [32, 42]
		yellow = [33, 43]
		blue = [34, 44]
		white = [37, 47]
	
	# 为了兼容之前写的代码
	black = color.black
	red = color.red
	green = color.green
	yellow = color.yellow
	blue = color.blue
	white = color.white
	
	@classmethod
	def print(cls, content: str, front_color: color, back_color: color = None):
		if back_color:
			print('\033[0;%2d;%2dm%s\033[0m' % (front_color.value[0], back_color.value[1], content))
		else:
			print('\033[%2dm%s\033[0m' %(front_color.value[0], content))


if __name__ == '__main__':
	Printer.print("Hello red", Printer.color.red)
	Printer.print("Hello green", Printer.color.green)
	Printer.print("Hello blue and white", Printer.blue, Printer.white)
	Printer.print('Hello red and blue', Printer.red, Printer.white)
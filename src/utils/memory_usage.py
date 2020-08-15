

import psutil


def memory():
	# gives a single float value
	psutil.cpu_percent()
	# gives an object with many fields
	psutil.virtual_memory()
	# you can convert that object to a dictionary 
	dict(psutil.virtual_memory()._asdict())
	# you can have the percentage of used RAM
	psutil.virtual_memory().percent


	# you can calculate percentage of available memory
	print("Available memory (%):", psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
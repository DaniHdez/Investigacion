import random

# para ejecucion 
#	$ python3 GeneradorDatos.py > datos.csv

def Generar(cantidad, rangoInicio, rangoFin):
	for i in range(0,cantidad):
		print("\"Dato\",",random.randint(rangoInicio+i, rangoFin+i))

Generar(700, 100, 200)
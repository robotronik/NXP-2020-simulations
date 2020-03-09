#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sqlite3
import sympy
import time

ENTRAXE = 0.2	# distance entre les 2 rotules de direction
RAYON = 0.02	# rayon des roues
THETA_MAX = 45.0*math.pi/180.0# angle de braquage maximum des roues en radian
L_AV = 0.1		# distance sur y entre le centre de gravite et l'essieu avant
L_AR = 0.1		# distance sur y entre le centre de gravite et l'essiu arriere
MASSE = 0.8		# masse de la voiture
J_INERTIE = 0.03# moment d'inertie de la voiture sur z au centre de gravite
G_TERRE = 9.81	# constante gravitationnelle sur terre
MU = 0.67		# coefficient (tan(phi)) de glissement roue/sol


class Voiture:
	"""
	represente la voiture
	"""
	def __init__(self):
		print("Initialisation of the voiture...")
		self.couple_arriere_droit = sympy.Symbol("c_arr_d")# couple du moteur sur la roue de droite
		self.couple_arriere_gauche = sympy.Symbol("c_arr_g")# couple du moteur sur la roue de gauche
		self.theta_direction = sympy.Symbol("theta_direction")# angle donne par le servo_moteur

		self.force_arriere_droit = {"x":sympy.Symbol("fx_arr_d"), "y":sympy.Symbol("fy_arr_d"), "z":None}# force du sol sur la roue arriere droite
		self.force_arriere_gauche = {"x":sympy.Symbol("fx_arr_g"), "y":sympy.Symbol("fy_arr_g"), "z":None}# force du sol sur la roue arriere gauche
		self.force_avant_droit = {"x":sympy.Symbol("fx_av_d"), "y":sympy.Symbol("fy_av_d"), "z":None}# force du sol sur la roue avant droite
		self.force_avant_gauche = {"x":sympy.Symbol("fx_av_g"), "y":sympy.Symbol("fy_av_g"), "z":None}# force du sol sur la roue avant gauche
		self.acceleration_translation = {"x":sympy.Symbol("ax_trans"), "y":sympy.Symbol("ay_trans")}# derivee temporelle de la vitesse de translation du veichule au centre d'inertie
		self.acceleration_angulaire = sympy.Symbol("omega_point_z")# acceleration engulaire selon z

		self.vitesses_translation = [{"x":0.0, "y":0.0}]# ancienne vitesses de translation de la voiture
		self.positions = [{"x":0.0, "y":0.0}]# les ancienne positions de la voiture
		self.vitesses_angulaire = [0.0]		# ancienne vitesses angulaires de la voiture
		self.angles = [0.0]					# les angles de la voiture
		self.t = [0.0]						# temps ecoule depuis le debut

		self.get_constant()					# cherche les valeurs numeriques qui remplacent les None
		self.U_vx, self.U_vy, self.U_omega_z = self.find_sequence()# expression des suites

		self.bdd = self.connect()
		self.cursor = self.bdd.cursor()
		print("\tfait!")

	def find_sequence(self):
		"""
		en fonction des derniers parametres calcules,
		cherche l'etat de la voiture dans 'dt' secondes
		retourne le nouvel etat de la voiture en fonction des etats precedents
		vx, vy et wz en fonction des couples arriere et de l'angle de direction
		"""
		def sign(x):
			"""
			'x' est un scalaire
			retourne le signe de x
			"""
			a = 100.0
			return 2.0*sympy.atan(a*x)/math.pi

		dt = sympy.Symbol("dt")
		inconues = [self.force_arriere_droit["x"], self.force_arriere_droit["y"],
					self.force_arriere_gauche["x"], self.force_arriere_gauche["y"],
					self.force_avant_droit["x"], self.force_avant_droit["y"],
					self.force_avant_gauche["x"], self.force_avant_gauche["y"],
					self.acceleration_angulaire,
					self.acceleration_translation["x"], self.acceleration_translation["y"],
					]

		# dynamique
		equation_moment = sympy.Eq(L_AR*(self.force_arriere_droit["x"] + self.force_arriere_gauche["x"])
			+ (ENTRAXE/2.0)*(self.force_arriere_droit["y"] - self.force_arriere_gauche["y"] + self.force_avant_droit["y"] - self.force_avant_gauche["y"])
			- (self.force_avant_droit["x"] + self.force_avant_gauche["x"])*L_AV
			, J_INERTIE*self.acceleration_angulaire)
		equation_acceleration_x = sympy.Eq(self.force_arriere_droit["x"] + self.force_arriere_gauche["x"]
			+ self.force_avant_droit["x"] + self.force_avant_gauche["x"]
			, MASSE*self.acceleration_translation["x"])
		equation_acceleration_y = sympy.Eq(self.force_arriere_droit["y"] + self.force_arriere_gauche["y"]
			+ self.force_avant_droit["y"] + self.force_avant_gauche["y"]
			, MASSE*self.acceleration_translation["y"])

		# glissement
		equation_gliss_d = sympy.Eq(self.force_arriere_droit["y"],
			sympy.Max(-MU*self.force_arriere_droit["z"], sympy.Min(MU*self.force_arriere_droit["z"], self.couple_arriere_droit/RAYON)))
		equation_gliss_g = sympy.Eq(self.force_arriere_gauche["y"],
			sympy.Max(-MU*self.force_arriere_gauche["z"], sympy.Min(MU*self.force_arriere_gauche["z"], self.couple_arriere_gauche/RAYON)))

		# pivot roue avant
		equation_pivot_droit = sympy.Eq(self.force_avant_droit["y"], sympy.tan(sympy.Min(THETA_MAX, sympy.Max(-THETA_MAX, self.theta_direction)))*self.force_avant_droit["x"])
		equation_pivot_gauche = sympy.Eq(self.force_avant_gauche["y"], sympy.tan(sympy.Min(THETA_MAX, sympy.Max(-THETA_MAX, self.theta_direction)))*self.force_avant_gauche["x"])

		# drift
		equation_drift_avd = sympy.Eq(self.force_avant_droit["x"], -sympy.cos(sympy.Min(THETA_MAX, sympy.Max(-THETA_MAX, self.theta_direction)))*self.force_avant_droit["z"]*MU*sign(
			sympy.cos(self.theta_direction)*(sympy.Symbol("vx") - L_AV*sympy.Symbol("omega_z"))
			+ sympy.sin(self.theta_direction)*(sympy.Symbol("vy") + (ENTRAXE/2.0)*sympy.Symbol("omega_z"))))
		equation_drift_avg = sympy.Eq(self.force_avant_gauche["x"], -sympy.cos(sympy.Min(THETA_MAX, sympy.Max(-THETA_MAX, self.theta_direction)))*self.force_avant_gauche["z"]*MU*sign(
			sympy.cos(self.theta_direction)*(sympy.Symbol("vx") - L_AV*sympy.Symbol("omega_z"))
			+ sympy.sin(self.theta_direction)*(sympy.Symbol("vy") - (ENTRAXE/2.0)*sympy.Symbol("omega_z"))))
		equation_drift_ard = sympy.Eq(self.force_arriere_droit["x"], -self.force_arriere_droit["z"]*MU*sign(sympy.Symbol("omega_z")*L_AR + sympy.Symbol("vx")))
		equation_drift_arg = sympy.Eq(self.force_arriere_gauche["x"], -self.force_arriere_gauche["z"]*MU*sign(sympy.Symbol("omega_z")*L_AR + sympy.Symbol("vx")))

		# resolution
		system = [equation_moment, equation_acceleration_x, equation_acceleration_y,
				equation_gliss_d, equation_gliss_g,
				equation_pivot_droit, equation_pivot_gauche,
				equation_drift_ard, equation_drift_arg, equation_drift_avd, equation_drift_avg]

		# [sympy.pprint(e) for e in system]

		resultats_sympy = sympy.solve(system, inconues)

		# print("*"*20)
		# [sympy.pprint(sympy.Eq(g,d)) for g,d in resultats_sympy.items()]

		resultat = {"omega_z":resultats_sympy[self.acceleration_angulaire]*dt + sympy.Symbol("omega_z")}
		resultat["vx"] = resultats_sympy[self.acceleration_translation["x"]]*dt \
						+ sympy.cos(resultat["omega_z"]*dt)*sympy.Symbol("vx") + sympy.sin(resultat["omega_z"]*dt)*sympy.Symbol("vy")
		resultat["vy"] = resultats_sympy[self.acceleration_translation["y"]]*dt \
						+ sympy.cos(resultat["omega_z"]*dt)*sympy.Symbol("vy") - sympy.sin(resultat["omega_z"]*dt)*sympy.Symbol("vx")
		return resultat["vx"], resultat["vy"], resultat["omega_z"]

	def find_folowing_numerical(self, delta_t, c_arr_d, c_arr_g, theta_direction):
		"""
		fait une application numerique des listes
		simule se qui se pace sur delta_t
		ne modifi pas le contenu des variables globales
		retourne l'etat absolu de la voiture (dans le referenciel du sol)
		"""
		t = time.time()
		erreur = "ABS(1000*(%s-delta_t)) + ABS(57*(%s-theta_direction)) + ABS(1000*(%s-c_arr_d)) + ABS(1000*(%s-c_arr_g)) + ABS(100*(%s-vx_in)) + ABS(100*(%s-vy_in)) + ABS(573*(%s-w_in))" % (
			delta_t, theta_direction, c_arr_d, c_arr_g, self.vitesses_translation[-1]["x"], self.vitesses_translation[-1]["y"], self.vitesses_angulaire[-1])
		self.cursor.execute("""
						SELECT vx_out, vy_out, w_out, x_out, y_out, angle, %s
						FROM simulation
						ORDER BY (%s)
						LIMIT 1""" % (erreur, erreur))
		resultat = list(self.cursor)
		if len(resultat):
			r = resultat[0]
			if resultat[0][6] < 0.2:
				resultat_num = {"vx":r[0], "vy":r[1], "omega_z":r[2],
					"x":r[3]+self.positions[-1]["x"], "y":r[4]+self.positions[-1]["y"], "angle":r[5]+self.angles[-1]}
				return resultat_num

		dt = min(2e-3, delta_t) # on ne fait pas des pas de plus de 2 ms
		dt = delta_t/float(int(delta_t/dt))
		resultat_num = {"vx":self.vitesses_translation[-1]["x"], "vy":self.vitesses_translation[-1]["y"], "omega_z":self.vitesses_angulaire[-1],
						"x":self.positions[-1]["x"], "y":self.positions[-1]["y"], "angle":self.angles[-1]}
		angle_relatif = 0.0
		x_relatif = 0.0
		y_relatif = 0.0
		resultat_form = {}
		resultat_form["vx"] = self.U_vx.subs({"dt":dt, "c_arr_d":c_arr_d, "c_arr_g":c_arr_g, "theta_direction":theta_direction})
		resultat_form["vy"] = self.U_vy.subs({"dt":dt, "c_arr_d":c_arr_d, "c_arr_g":c_arr_g, "theta_direction":theta_direction})
		resultat_form["omega_z"] = self.U_omega_z.subs({"dt":dt, "c_arr_d":c_arr_d, "c_arr_g":c_arr_g, "theta_direction":theta_direction})
		for i in range(int(delta_t/dt)):
			resultat_num = {
				"vx":resultat_form["vx"].subs({"vx":resultat_num["vx"], "vy":resultat_num["vy"], "omega_z":resultat_num["omega_z"]}),
				"vy":resultat_form["vy"].subs({"vx":resultat_num["vx"], "vy":resultat_num["vy"], "omega_z":resultat_num["omega_z"]}),
				"omega_z":resultat_form["omega_z"].subs({"vx":resultat_num["vx"], "vy":resultat_num["vy"], "omega_z":resultat_num["omega_z"]}),
			}
			angle_relatif += dt*resultat_num["omega_z"]
			x_relatif += dt*(resultat_num["vx"]*math.cos(self.angles[-1]+angle_relatif) - resultat_num["vy"]*math.sin(self.angles[-1]+angle_relatif))
			y_relatif += dt*(resultat_num["vx"]*math.sin(self.angles[-1]+angle_relatif) + resultat_num["vy"]*math.cos(self.angles[-1]+angle_relatif))


		self.cursor.execute("""INSERT INTO simulation
			(delta_t, theta_direction, c_arr_d, c_arr_g, vx_in, vy_in, w_in, vx_out, vy_out, w_out, x_out, y_out, angle)
			VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""" %
			(delta_t, theta_direction, c_arr_d, c_arr_g, self.vitesses_translation[-1]["x"], self.vitesses_translation[-1]["y"], self.vitesses_angulaire[-1],
				resultat_num["vx"], resultat_num["vy"], resultat_num["omega_z"], x_relatif, y_relatif, angle_relatif))
		self.bdd.commit()

		resultat_num["x"] = self.positions[-1]["x"] + x_relatif
		resultat_num["y"] = self.positions[-1]["y"] + y_relatif
		resultat_num["angle"] = self.angles[-1] + angle_relatif
		return resultat_num

	def find_consignes(self, x_abs, y_abs, vx_abs, vy_abs):
		"""
		cherche les parametres c_arr_d, c_arr_g et l'angle des roues
		afin d'ateindre au mieu l'objectif.
		"""
		def erreur(x_abs, y_abs, vx_abs, vy_abs, c_arr_d, c_arr_g, theta_direction):
			"""
			retourne un scalaire positif qui represente la somme des carres des ecarts entre:
				-la plus courte distance entre la trajectoire de la voiture et le point (x_abs, y_abs) [en m**2]
					-pondere par 'coef_dist' [en 1/m**2]
				-le temps mis pour atteindre ce point [en s]
					-pondere par 'coef_temps' [en 1/s]
				-l'ecart angulaire entre le vecteur vitesse desire et celui obtenu [en rad]
					-pondere par 'coef_angle' [en 1/rad]
				-l'ecart entre la vitesse desiree en norme et la vitesse atteinte [en m/s]
					-pondere par 'coef_vitesse' [en s/m]
			"""
			def find_t(f, t_min, t_max):
				if t_max - t_min < 0.04e-3:
					return t_max
				n = 100
				meilleur_t = t_min
				valeur = f(t_min)
				for i in range(1, n):
					t = i*(t_max-t_min)/(n-1) + t_min
					actuel = f(t)
					if actuel < valeur:
						valeur = actuel
						meilleur_t = t
				pas = (t_max-t_min)/(n-1)
				return find_t(f, max(t_min, meilleur_t-pas), min(t_max, meilleur_t+pas))
			
			coef_dist = 1.0
			coef_temps = 10.0
			coef_angle = 0.3
			coef_vitesse = 0.1

			delta_t = 100e-3
			while 1:
				r = self.find_folowing_numerical(delta_t, c_arr_d, c_arr_g, theta_direction)
				vx1 = r["vx"]*math.cos(r["angle"]) - r["vy"]*math.sin(r["angle"])
				vy1 = r["vy"]*math.cos(r["angle"]) + r["vx"]*math.sin(r["angle"])
				ax = 0					# x0
				ay = 0					# y0
				bx = self.vitesses_translation[-1]["x"]# vx0
				by = self.vitesses_translation[-1]["y"]# vy0
				cx = (-delta_t*(2*self.vitesses_translation[-1]["x"] + vx1) - 3*self.positions[-1]["x"] + 3*r["x"])/delta_t**2# (-dt*(2*vx0 + vx1) - 3*x0 + 3*x1)/dt**2
				cy = (-delta_t*(2*self.vitesses_translation[-1]["y"] + vy1) - 3*self.positions[-1]["y"] + 3*r["y"])/delta_t**2# (-dt*(2*vy0 + vy1) - 3*y0 + 3*y1)/dt**2
				dx = (delta_t*(self.vitesses_translation[-1]["x"] + vx1) + 2*self.positions[-1]["x"] - 2*r["x"])/delta_t**3# (dt*(vx0 + vx1) + 2*x0 - 2*x1)/dt**3
				dy = (delta_t*(self.vitesses_translation[-1]["y"] + vy1) + 2*self.positions[-1]["y"] - 2*r["y"])/delta_t**3# (dt*(vy0 + vy1) + 2*y0 - 2*y1)/dt**3

				distance = lambda t: (ax+bx*t+cx*t**2+dx*t**3-x_abs)**2 + (ay+by*t+cy*t**2+dy*t**3-y_abs)**2
				t = find_t(distance, 0, delta_t)
				if delta_t - t <= 0.1e-3:
					delta_t *= 2
					continue
				print("t:",t)
				vx = bx + 2*cx*t + 3*dx*t**2
				vy = by + 2*cy*t + 3*dy*t**2
				norme = math.sqrt(vx**2 + vy**2)

				erreur = coef_dist*distance(t) \
						+ coef_temps*t \
						+ coef_angle*abs(vx*vy_abs - vy*vx_abs)/(norme*math.sqrt(vx_abs**2 + vy_abs**2)) \
						+ coef_vitesse*norme

				T = [delta_t*i/1000 for i in range(1000)]
				X = [ax+bx*t+cx*t**2+dx*t**3 for t in T]
				Y = [ay+by*t+cy*t**2+dy*t**3 for t in T]
				plt.plot(X, Y)
				plt.axis("equal")
				plt.draw()
				plt.pause(0.01)
				return erreur

		plt.scatter(x_abs, y_abs)
		for c_arr_d_e in range(-10, 11):
			c_arr_d = 0.03*c_arr_d_e/10
			for c_arr_g_e in range(-10, 11):
				c_arr_g = 0.03*c_arr_g_e/10
				for theta_e in range(-10, 11):
					theta = 30*math.pi/180*theta_e/10
					erreur(x_abs, y_abs, vx_abs, vy_abs, c_arr_d, c_arr_g, theta)

	def get_constant(self):
		"""
		calcule tous les parametres invariant et les ajoute
		au variables 'globales'
		"""
		force_essieu_avant = MASSE*G_TERRE*L_AR/(L_AR + L_AV) # regle des moments
		force_essieu_arriere = MASSE*G_TERRE*L_AV/(L_AR + L_AV)# en faisant l'hypotese que la voiture est tres deformable
		self.force_arriere_droit["z"] = force_essieu_arriere/2.0
		self.force_arriere_gauche["z"] = force_essieu_arriere/2.0
		self.force_avant_droit["z"] = force_essieu_avant/2.0
		self.force_avant_gauche["z"] = force_essieu_avant/2.0

	def simulation(self, t_max, c_arr_d, c_arr_g, theta_direction):
		"""
		simule la trajectoire de la voiture
		en fonction de c_arr_d(t), c_arr_g(t), theta_direction(t)
		"""
		delta_t = 0.05 # 50 ms
		t = 0
		i = 0
		plt.title("Trajectoire de la voiture: 1 pt toutes les %dms" % int(1000*delta_t))
		plt.xlabel("position (m)")
		plt.ylabel("position (m)")
		last_time = time.time()
		while t < t_max:
			print("t=%f, V=%s, W=%s" % (t, self.vitesses_translation[-1], self.vitesses_angulaire[-1]))
			plt.scatter(self.positions[-1]["x"], self.positions[-1]["y"])
			plt.arrow(self.positions[-1]["x"], self.positions[-1]["y"], -0.01*math.sin(self.angles[-1]), 0.01*math.cos(self.angles[-1]),
						head_width=0.04)
			if time.time()-last_time > 0:
				last_time += 30
				plt.axis("equal")
				plt.savefig("simulation_nxp.svg")
				plt.draw()
				plt.pause(0.01)
			r = self.find_folowing_numerical(delta_t=delta_t, c_arr_d=c_arr_d(t), c_arr_g=c_arr_g(t), theta_direction=theta_direction(t))
			self.vitesses_angulaire.append(r["omega_z"])
			self.vitesses_translation.append({"x":r["vx"], "y":r["vy"]})
			self.angles.append(r["angle"])
			self.positions.append({"x":r["x"], "y":r["y"]})
			t+=delta_t
			i+=1

	def connect(self):
		"""
		retourne une base de donne qui contient les bonnes infos
		"""
		name = "ENTRAXE:%g_RAYON:%g_L_AV:%g_L_AR:%g_MASSE:%g_J_INERTIE:%g_MU:%g.db" % (ENTRAXE, RAYON, L_AV, L_AR, MASSE, J_INERTIE, MU)
		bdd = sqlite3.connect(name)
		curseur = bdd.cursor()
		curseur.execute("SELECT name FROM sqlite_master WHERE type='table';")
		liste = curseur.fetchall()
		if liste == [("simulation",)]:
			return bdd
		else:
			bdd.execute("""CREATE TABLE simulation
				(id INTEGER PRIMARY KEY,
				delta_t DOUBLE,
				theta_direction DOUBLE,
				c_arr_d DOUBLE,
				c_arr_g DOUBLE,
				vx_in DOUBLE,
				vy_in DOUBLE,
				w_in DOUBLE,
				vx_out DOUBLE,
				vy_out DOUBLE,
				w_out DOUBLE,
				x_out DOUBLE,
				y_out DOUBLE,
				angle DOUBLE)""")
			bdd.commit()
		return bdd

class Reader:
	"""
	permet de lire un fichier txt de contenu:
		camera 55 ... 27
		camera 124 ... 12
		...
		camera 38 23 ... 43
	"""
	def __init__(self, *paths):
		self.paths = paths						# nom du fichier txt a ouvrir

	def __iter__(self):
		"""
		permet la syntaxe:
			for matrice in self:
		"""
		for filename in self.get_txt_name(*self.paths):
			try:
				with open(filename, "r", encoding="utf-8") as f:
					for ligne in f:
						pixels = [int(nombre) for nombre in ligne.split(" ") if nombre.isdigit()]
						if len(pixels) == 128:
							yield pixels
			except UnicodeDecodeError:
				pass

	def get_txt_name(self, *paths):
		"""
		retourne les chemins de tous les fichiers txt
		"""
		for path in paths:
			if path == "":
				path = os.getcwd()
			if os.path.isfile(path):
				if path[-4:].lower() == ".txt":
					yield path
			elif os.path.isdir(path):
				for papa, dossiers, files in os.walk(path):
					for file in self.get_txt_name(*[os.path.join(papa, f) for f in files]):
						yield file

class Analyser:
	"""
	tente d'interpreter la camera
	"""
	def __init__(self, iterateur):
		self.iterateur = iterateur # objet qui cede les pixels en brute de la camera

		self.figure = plt.figure(figsize=(14, 14), dpi=70)# x, y
		self.figure.suptitle("Analyse Independante")
		grille_spec = self.figure.add_gridspec(5, 1) # y, x
		self.ax1 = self.figure.add_subplot(grille_spec[0:2, :])
		self.ax2 = self.figure.add_subplot(grille_spec[2:4, :])
		self.ax3 = self.figure.add_subplot(grille_spec[4, :])

	def analyse_une_seule_image(self, pixels):
		"""
		fait une analyse d'une seule image
		retourne le maximum d'informations que l'on puisse extraire
		avec une seule image
		"""
		def find_polynome(pixels):
			"""
			cherche le polinome d'ordre 2 qui permet au mieu
			de normaliser la courbe
			le polinome est de la forme a*x**2+b*x+c
			"""
			def diff(f, x):
				"""
				derive f en x
				"""
				h = 1e-8
				return (f(x+h)-f(x-h))/(2*h)

			def distance(a, b, c, pixels, rayon):
				"""
				retourne la distance entre les pixels et le polinome
				c'est comme un moindre carre mais en plus subtile
				"""
				d = 0
				for x, pixel in enumerate(pixels):
					d_locale = abs(pixel - (a*(x-64)**2 + b*(x-64) + c))
					if d_locale < rayon:
						d += d_locale
				return d

			def gradient(a, b, c, pixels, rayon):
				"""
				retourne la derivee partiel de l'erreur selon les trois
				composante a, b et c
				"""
				return (
				diff(lambda a: distance(a=a, b=b, c=c, pixels=pixels, rayon=rayon), a),
				diff(lambda b: distance(a=a, b=b, c=c, pixels=pixels, rayon=rayon), b),
				diff(lambda c: distance(a=a, b=b, c=c, pixels=pixels, rayon=rayon), c),
				)

			def init(pixels):
				"""
				retourne le a, le b et le c initial
				"""
				haut = sum(pixels[64:])/64
				bas = sum(pixels[:64])/64
				a = -0.001
				b = (haut-bas)/128
				c = .5*(bas+haut)
				return a, b, c

			a, b, c = init(pixels)

			rayon = 100
			i = 0
			while rayon >= 7:
				i+=1
				rayon /= 1.004
				grad = gradient(a, b, c, pixels, rayon)
				a -= 1e-8 * grad[0]
				b -= 3e-6 * grad[1]
				c -= 1e-3 * grad[2]

			return a, b, c

		def normalize(a, b, c, pixels):
			"""
			retourne la liste des points entre -1 et 1
			"""
			difference = [p - a*(i-64)**2 - b*(i-64) - c for i,p in enumerate(pixels)]
			maximum = max(difference)
			minimum = -min(difference)
			borne = max(maximum, minimum)
			if borne:
				image_redressee = [d/borne for d in difference]
			else:
				image_redressee = [0 for d in difference]
			return image_redressee

		def get_seuil(image_redressee):
			"""
			retourne un scalaire entre -1 et 1
			ce scalaire est tel que tous ce qui est en dessous
			est noir et tous ce qui est au dessu est blanc
			"""
			pourcentage = 0.1 # pourcentage de noir dans l'image
			bordure = 0.05 # pour le calcul du seuil 'median', 0 pour prendre le extremitee, 0.5 pour prendre le centre
			poid_pourcentage_seuil = 0.4 # influence du pourcentage 'seuil_pourcentage'. 0: aucune influence, 1: seul lui compte
			pixels_triees = sorted(image_redressee)
			seuil_pourcentage = pixels_triees[int(pourcentage*len(image_redressee))] # c'est le seuil tel que pourceantage des points soient en dessous de ce seuil
			seuil_median = (pixels_triees[int(len(image_redressee)*bordure)]+pixels_triees[-int(len(image_redressee)*bordure)])/2

			return seuil_median*(1-poid_pourcentage_seuil) + seuil_pourcentage*poid_pourcentage_seuil

		def get_probas_couleur(image_redressee, seuil):
			"""
			retourne pour chaque pixel la probabilite qu'il soit blanc
			0: noir certain
			0.5: couci-coussa
			1: blanc certain
			"""
			ordonne = sorted(((rang_initial, ecart) for rang_initial, ecart in enumerate(image_redressee)), key=(lambda couple: couple[1]))# liste de tous les points classes par ordre croissant
			rang_seuil = sorted([(faux_rang, abs(ecart-seuil)) for faux_rang, (rang_initial, ecart) in enumerate(ordonne)], key=(lambda couple: couple[1]))[0][0]# on recupere le rang de la liste ordonne ou se trouve le suile
			rang_initial_couleur = [(rang_initial,
									rang*0.5/rang_seuil
									if rang <= rang_seuil else
									0.5 + (rang-rang_seuil)*0.5/(len(ordonne)-1-rang_seuil)
									) for rang, (rang_initial, ecart) in enumerate(ordonne)]
			probas_couleur = [proba for rang_initial, proba in sorted(rang_initial_couleur, key=lambda couple: couple[0])]
			return probas_couleur

		def get_masse_zonnes(probas_couleur):
			"""
			retourne une liste de la meme longueur que 'probas_couleur'
			repere chaque zonnes (noir ou blanche)
			dans chaque zonne, l'aire algebrique entre 0,5 et probas_couleur
			est presente de partout
			la valeur renvoyee est normee entre 0 et 1
			"""
			ALPHA = 2.0 # entre 0 et oo, oo => masse_relative = sign(masse_absolu)

			somme_cumul = [0]
			for proba_couleur in probas_couleur:
				if proba_couleur >= 0.5 and somme_cumul[-1] >= 0 \
				or proba_couleur < 0.5 and somme_cumul[-1] < 0:
					somme_cumul.append(somme_cumul[-1] + proba_couleur - 0.5)
				else:
					somme_cumul.append(proba_couleur - 0.5)
			del somme_cumul[0]

			masse_absolu = [somme_cumul[-1]]
			for i in range(len(somme_cumul)-2, -1, -1):
				if somme_cumul[i]*masse_absolu[0] > 0:							# si c'est toujour du meme signe
					masse_absolu.insert(0, masse_absolu[0])						# on met a plat
				else:															# si le signe change
					masse_absolu.insert(0, somme_cumul[i])						# on repart a zero

			masse_relative = list(map(lambda p: 0.5 + math.atan(ALPHA*p)/math.pi, masse_absolu))

			return masse_relative

		# calculs
		resultat = {}
		resultat["luminosite_moyenne"] = sum(pixels)/128						# cette grandeur permet d'adapter le temps avant de prendre l'image suivante
		resultat["a"], resultat["b"], resultat["c"] = find_polynome(pixels)		# le a, b, c initial
		resultat["image_redressee"] = normalize(resultat["a"], resultat["b"], resultat["c"], pixels)# l'image comprise entre 0 et 1
		resultat["seuil"] = get_seuil(resultat["image_redressee"])				# le seuil qui separe blanc/noir
		resultat["probas_couleur"] = get_probas_couleur(resultat["image_redressee"], resultat["seuil"])# probabilite qui donne la couleur du sol
		resultat["masse_zonnes"] = get_masse_zonnes(resultat["probas_couleur"])	# l'aire de chaque 'bloc'

		# affichage
		self.ax1.cla()
		self.ax1.axis((-64, 64, 0, 255))
		self.ax1.set_title("Image Brute")
		self.ax1.plot(list(range(-64, 64)), pixels, label="camera brute")
		self.ax1.plot(list(range(-64, 64)), [resultat["a"]*x**2 + resultat["b"]*x + resultat["c"] for x in range(-64, 64)], label="polynome")

		self.ax2.cla()
		self.ax2.axis((-64, 64, -1.1, 1.1))
		self.ax2.set_title("Image Redressee")
		self.ax2.plot(list(range(-64, 64)), resultat["image_redressee"], label="camera formatte")
		self.ax2.plot([-64, 64], [resultat["seuil"], resultat["seuil"]], label="seuil")

		self.ax3.cla()
		self.ax3.axis((-64, 64, -.1, 1.1))
		self.ax3.set_title("Couleur Finale")
		self.ax3.plot([-64, 64], [0.5, 0.5])
		self.ax3.plot(list(range(-64, 64)), resultat["probas_couleur"], label="hors de tout contexte")
		self.ax3.plot(list(range(-64, 64)), resultat["masse_zonnes"], label="masse hors contexte")

		self.ax1.legend()
		self.ax2.legend()
		self.ax3.legend()
		plt.draw()
		plt.pause(0.01)

		return resultat

	def positionne_dans_espace(self, matrice_pixels, vitesses_lineaire, vitesses_angulaire, dt, theta0):
		"""
		'matrice_pixels' est une liste de liste de pixels
		'vitesses_lineaire' est une liste de la norme des vitesses au centre de gravite de la voiture
		'vitesses_angulaire' est une liste des vitesses de rotation sur l'axe vertical de la voirure
		'dt' est l'intevalle de temps entre chaque images
		'theta0' est l'angle initial en radian que forme la voiture par raport a l'axe de la piste
		"""
		def somme_cumulee(liste):
			s = 0.0
			out = []
			for e in liste:
				s += e
				out.append(s)
			return out

		assert len(matrice_pixels) == len(vitesses_lineaire) == len(vitesses_angulaire), "Tous les elements doivent avoir la meme taille"
		assert len(matrice_pixels) >= 2, "Il doit y avoir au moins 2 images pour faire une analyse"

		angles_voiture = list(map(
			lambda theta : (cos(theta + theta0), sin(theta + theta0)),
				[0] + somme_cumulee([dt*(vitesses_angulaire[i] + vitesses_angulaire[i+1])/2
				for i in range(len(vitesses_angulaire) - 1)])
			)) # a chaque image, associ le vecteur (x, y) de la direction de la voiture, ce vecteur est norme



	def analyse_multiple(self, pixels, ancienne_proba):
		"""
		pixels est la liste des pixels brutes issuent de la camera
		'ancienne_proba' donne la proba dans chaque zonne
		"""
		resultat = {}	# initialisation du resultat
		maintenant = self.analyse_une_seule_image(pixels)	# resultat ne depandant que de cette image

		# # recherche des probas
		# maximum = max(maintenant["zonne_brute"])
		# quoeff = 0.8 # poid de l'ancienne image (entre 0 et 1)
		# probas_pas_norme = [((1-quoeff)*z/maximum + quoeff*abs(p)) for z,p in zip(maintenant["zonne_brute"], ancienne_proba)] # on prend 100a% la proba de l'ancienne image et 100-100a % la nouvelle image
		# maximum = max(probas_pas_norme)
		# probas_presence = [p/maximum for p in probas_pas_norme]# compris entre 0 et 1

		# # probas finales
		# quoeff = 100 # coefficient multiplicateur pour faire presque saturer la difference entre 1 et beaucoup
		# norme_triee = sorted(maintenant["norme"])
		# seuil = (norme_triee[5]+norme_triee[-5])/2
		# self.ax2.plot([-64, 64], [seuil, seuil], label="Seuil De Comparaison")
		# resultat["probas_couleurs"] = [max(-1, min(1, quoeff*pp*(pn-seuil))) for pp, pn in zip(probas_presence, maintenant["norme"])]

		# couleurs
		# couleurs = [1 if p >= 0 else 0 for p in resultat["probas_couleurs"]]	# on fait une comparaison grossiere

		# resultat["couleurs"] = couleurs
		# self.ax3.cla()
		# self.ax3.axis(xmin=-64, xmax=64)
		# self.ax3.set_title("Couleur Finale")
		# self.ax3.plot(list(range(-64, 64)), maintenant["couleurs"], label="1: Blanc, 0: Noir")

		# return resultat
		return {"probas_couleur":maintenant["probas_couleur"]}

	def __iter__(self):
		"""
		cede a chaque iteration les couleurs du terrain
		"""
		probas = None
		for pixels in self.iterateur: # pour chaque image de la camera
			if probas is None:
				probas = [1 for p in pixels]
			resultat = self.analyse_multiple(pixels, probas)
			yield resultat
			probas = resultat["probas_couleur"]

			# self.ax3.plot(list(range(-64, 64)), probas, label="Couleur Du Sol")

			# self.ax1.legend()
			# self.ax2.legend()
			# self.ax3.legend()
			# plt.draw()
			# plt.pause(0.01)


# for r in Analyser(Reader("")):
# 	pass

nxp = Voiture()
# sympy.pprint(nxp.simulation(
# 	t_max=3600,
# 	c_arr_d=lambda t:.01,
# 	c_arr_g=lambda t:.01 if t < 60 else -.01 if t<60.5 else 0.01,
# 	theta_direction=lambda t:0 if t<60 else 5*math.pi/180 if t<60.5 else 0))
nxp.find_consignes(0, 0.1, 1, 0)

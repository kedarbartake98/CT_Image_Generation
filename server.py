from flask import Flask, render_template, url_for, jsonify, request
import pandas as pd
import os
import random, json
import config
import sys
import numpy as np
from PCA_rep import PCA_rep
import cv2

IMG_SOURCES = {	
	'source_img': 'static/images/source.png',
	'dest_img': 'static/images/dest.png',
	'interpolated': 'static/images/interpolated.png'
}

class Images:
	
	def __init__(self):
		self.source = None
		self.dest = None 
		self.interpolated = None

	def clear(self):
		self.source = None
		self.dest = None 
		self.interpolated = None

	def set_new_src_dest(self):
		self.clear()
		src_img, src_sample = pca.generate_random_image_sample()
		self.source = (src_img, src_sample)
		print('Writing')
		cv2.imwrite(IMG_SOURCES['source_img'], src_img)

		dst_img, dst_sample = pca.generate_random_image_sample()
		self.dest = (dst_img, dst_sample)
		cv2.imwrite(IMG_SOURCES['dest_img'], dst_img)		

	def interpolate(self, inc):

		self.interpolated = pca.interpolate(self.source[1], self.dest[1], inc)
		cv2.imwrite(IMG_SOURCES['interpolated'], self.interpolated)

app = Flask(__name__)
pca = PCA_rep(config)
image = Images()

@app.route('/')
def entrypoint():
	return render_template('index.html')

@app.route('/generate_images', methods=['POST'])

def generate():
	print('generate called')
	image.set_new_src_dest()
	return {'source' : url_for('static', filename='images/source.png'),
			'dest' : url_for('static', filename="images/dest.png"),
			'interpolated': url_for('static',  filename='images/source.png')}

@app.route('/interpolate', methods=['POST'])

def interpolate():
	inc = request.form['inc']
	print('Inc', inc)
	image.interpolate(float(inc))
	return {'interpolated': url_for('static', filename="images/interpolated.png")}

if __name__=='__main__':
	app.run(debug=True)
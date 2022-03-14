#!/usr/bin/env python3

#################################
# Example interactive fitting GUI using PySimpleGUI
# Fits a sine wave with noise 
# Author: David Gallacher
# Date: March 2022
#################################


import PySimpleGUI as sg
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Note the matplot tk canvas import
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

#globals
_VARS = {'window': False,'fig_agg':False,'pltFig':False}
AppFont = 'Any 16'
sg.theme('LightBlue')
linecolor = '#4062c9' #match dark amber theme

#Plot default parameters
pltPars = {
	'nSamples' : 1000,
	'low' : 0,
	'high' : 5,
	'amp' : 10,
	'phase' : 1,
	'offset' : 0.0,
	'freq' : 1,
	'sd' : 12345, # Seed of PRNG
	'smear':True,
	'smear_mu' : 1,
	'smear_sig' : 1/3
}

#Default fit parameters
fitVars = {
	'amp': 1.0,
	'phase': 1.0,
	'offset': 0.0,
	'freq' : 1.0,
}

#Fit limits not implemented
#Limits given as a 2-tuple of arrays with size = Npars EG: ([low1,low2,...,lowN],[high1,high2,...highN])
fitLimits = ([0,0,0,0],[0,0,0,0])


#Function to plot, y = A*sin(w*x+phi) + C
def SineWaveFunc(x,_amp,_phase,_freq,_offset):
	return _amp*np.sin(2*np.pi*_freq*x + np.full_like(x,np.pi/2.0*_phase)) + _offset

#Method to fit data, return fitted curve, fitted parameters and covariance matrix
def tryFit(_x,_y,_starting,_bounds):
	popt, pcov = curve_fit(SineWaveFunc,_x,_y,p0=_starting)
	fitYvalues = SineWaveFunc(_x,*popt)
	return fitYvalues,popt,pcov


class Toolbar(NavigationToolbar2Tk):
	def ___init__(self,*args,**kwargs):
		super(Toolbar,self).__init(*args,**kwargs)


def draw_figure_w_toolbar(canvas,figure,canvas_toolbar):
	if canvas.children:
		for child in canvas.winfo_children():
			child.destroy()
	if canvas_toolbar.children:
		for child in canvas_toolbar.winfo_children():
			child.destroy()

	figure_canvas_agg = FigureCanvasTkAgg(figure, master=canvas)
	figure_canvas_agg.draw()
	toolbar = Toolbar(figure_canvas_agg,canvas_toolbar)
	toolbar.update()
	figure_canvas_agg.get_tk_widget().pack(side='left', fill='both', expand=1)
	return figure_canvas_agg


#initialize plot with defaults
def makePlot():
	_VARS['pltFig'] = plt.gcf()

	#Make the data to plot
	xData = np.linspace(pltPars['low'],pltPars['high'],num = pltPars['nSamples'])	
	yData = SineWaveFunc(xData,pltPars['amp'],pltPars['phase'],pltPars['freq'],pltPars['offset'])
	#Add gaussian smearing, proportional to amplitude
	np.random.seed(pltPars['sd'])
	#Check before smearing
	if pltPars['smear']: 
		yData = yData + np.random.normal(loc=pltPars['smear_mu'], scale=pltPars['smear_sig'], size=pltPars['nSamples'])

	plt.figure(1)
	DPI = _VARS['pltFig'].get_dpi()
	_VARS['pltFig'].set_size_inches(404*3/float(DPI),404/float(DPI))
	frameMain = _VARS['pltFig'].add_axes((.1,.3,.8,.6))
	#ax = _VARS['pltFig'].add_subplot(111)
	plt.plot(xData,yData,color=linecolor)
	plt.ylabel('Amplitude [au]')
	plt.grid()
	frameMain.set_xticklabels([]) #Remove x-tic labels for the first frame

	frameRes = _VARS['pltFig'].add_axes((.1,.1,.8,.2))  
	plt.plot(xData,np.zeros_like(xData),'r')
	plt.grid()
	plt.xlabel('Time [s]')
	plt.ylabel('Residuals [(Data-Fit)/Data]',fontsize='x-small')

	_VARS['fig_agg'] = draw_figure_w_toolbar(_VARS['window']['figCanvas'].TKCanvas,_VARS['pltFig'],_VARS['window']['controls_cv'].TKCanvas)


#Clear old plot and make a new one
def updatePlot():
	_VARS['fig_agg'].get_tk_widget().forget()
	_VARS['pltFig'].clf()
	xData = np.linspace(pltPars['low'],pltPars['high'],num = pltPars['nSamples'])
	yData = SineWaveFunc(xData,pltPars['amp'],pltPars['phase'],pltPars['freq'],pltPars['offset'])
	#Add gaussian smearing, proportional to amplitude
	np.random.seed(pltPars['sd'])
	#Check before smearing
	if pltPars['smear']: 
		yData = yData+np.random.normal(loc=pltPars['smear_mu'],scale=pltPars['smear_sig'],size=pltPars['nSamples'])
	plt.figure(1)
	DPI = _VARS['pltFig'].get_dpi()
	_VARS['pltFig'].set_size_inches(404*3/float(DPI),404/float(DPI))
	frameMain = _VARS['pltFig'].add_axes((.1,.3,.8,.6))
	plt.plot(xData,yData,color=linecolor,label='Simulated Data')
	plt.ylabel('Amplitude [au]')
	plt.grid()

	fitResult = np.zeros_like(xData) 
	residuals = (fitResult-yData)
	plt.plot(xData,fitResult,'r--',label='Fit result')
	plt.legend(loc='upper right')

	frameMain.set_xticklabels([]) #Remove x-tic labels for the first frame
	frameRes = _VARS['pltFig'].add_axes((.1,.1,.8,.2))  
	plt.plot(xData,residuals,'r')
	plt.grid()
	plt.xlabel('Time [s]')
	plt.ylabel('Residuals [Data-Fit]',fontsize='x-small')

	_VARS['fig_agg'] = draw_figure_w_toolbar(_VARS['window']['figCanvas'].TKCanvas,_VARS['pltFig'],_VARS['window']['controls_cv'].TKCanvas)


#Replot with new fit 
def updatePlotWithFit():
	_VARS['fig_agg'].get_tk_widget().forget()
	_VARS['pltFig'].clf()
	xData = np.linspace(pltPars['low'],pltPars['high'],num = pltPars['nSamples'])
	yData = SineWaveFunc(xData,pltPars['amp'],pltPars['phase'],pltPars['freq'],pltPars['offset'])
	np.random.seed(pltPars['sd'])
	#Check before smearing
	if pltPars['smear']: 
		yData = yData+np.random.normal(loc=pltPars['smear_mu'],scale=pltPars['smear_sig'],size=pltPars['nSamples'])
	plt.figure(1)
	DPI = _VARS['pltFig'].get_dpi()
	_VARS['pltFig'].set_size_inches(404*3/float(DPI),404/float(DPI))
	frameMain = _VARS['pltFig'].add_axes((.1,.3,.8,.6))
	plt.plot(xData,yData,color=linecolor,label='Simulated Data')
	plt.ylabel('Amplitude [au]')
	plt.grid()

	#Updated Fit
	initParams = [fitVars['amp'],fitVars['phase'],fitVars['freq'],fitVars['offset']] #Get fit params from dictionary
	fitResult, fitPars,fitCov = tryFit(xData,yData,initParams,fitLimits)
	residuals = (fitResult-yData)

	plt.plot(xData,fitResult,'r--',label='Fit result')
	plt.legend(loc='upper right')
	frameMain.set_xticklabels([]) #Remove x-tic labels for the first frame
	frameRes = _VARS['pltFig'].add_axes((.1,.1,.8,.2))  
	plt.plot(xData,residuals,'r')
	plt.grid()
	plt.xlabel('Time [s]')
	plt.ylabel('Residuals [Data-Fit]',fontsize='x-small')

	#show RSS of fit
	rss = getRSS(fitResult,yData)
	_VARS['window']['_fit_gf'].update(str(rss)[:6])#Only show first six figs.
	_VARS['window']['_fit_amp'].update(str(fitPars[0])[:6])
	_VARS['window']['_fit_phase'].update(str(fitPars[1])[:6])
	_VARS['window']['_fit_freq'].update(str(fitPars[2])[:6])
	_VARS['window']['_fit_offset'].update(str(fitPars[3])[:6])

	_VARS['fig_agg'] = draw_figure_w_toolbar(_VARS['window']['figCanvas'].TKCanvas,_VARS['pltFig'],_VARS['window']['controls_cv'].TKCanvas)


#Get RSS of fit to display
def getRSS(fitY,dataY):
	RSS = 0.0
	for i, val in enumerate(fitY):
		RSS = RSS + (dataY[i]-val)**2
	return RSS


#Get input data from GUI and set fit initial parameters
def updateFitVars(vals):
	#Check that input values aren't empty
	if vals['fit_amp']:
		_VARS['window']['_fit_amp'].update(vals['fit_amp'])
		fitVars['amp'] = float(vals['fit_amp'])
	if vals['fit_phase']:
		_VARS['window']['_fit_phase'].update(vals['fit_phase'])
		fitVars['phase'] = float(vals['fit_phase'])
	if vals['fit_freq']:
		_VARS['window']['_fit_freq'].update(vals['fit_freq'])
		fitVars['freq'] = float(vals['fit_freq'])
	if vals['fit_offset']:
		_VARS['window']['_fit_offset'].update(vals['fit_offset'])
		fitVars['offset'] = float(vals['fit_offset'])

#Method to update pltPars dictionary with input data
# also display updated input data 
def updateVars(vals):
	
	#Check that input values aren't empty
	if vals['nsamp']:
		_VARS['window']['_nsamp'].update(vals['nsamp'])
		pltPars['nSamples'] = int(vals['nsamp'])
	if vals['low']:
		_VARS['window']['_low'].update(vals['low'])
		pltPars['low'] = int(vals['low'])
	if vals['high']:
		_VARS['window']['_high'].update(vals['high'])
		pltPars['high'] = int(vals['high'])
	if vals['amp']:
		_VARS['window']['_amp'].update(vals['amp'])
		pltPars['amp'] = float(vals['amp'])
	if vals['phase']:
		_VARS['window']['_phase'].update(vals['phase'])
		pltPars['phase'] = float(vals['phase'])
	if vals['offset']:
		_VARS['window']['_offset'].update(vals['offset'])
		pltPars['offset'] = float(vals['offset'])
	if vals['freq']:
		_VARS['window']['_freq'].update(vals['freq'])
		pltPars['freq'] = float(vals['freq'])
	if vals['sd']:
		_VARS['window']['_sd'].update(vals['sd'])
		pltPars['sd'] = int(vals['sd'])
	if vals['smear']:
		_VARS['window']['_smear'].update(vals['smear'])
		pltPars['smear'] = (vals['smear']).lower() in ("t","true","1")
	if vals['smear_mu']:
		_VARS['window']['_smear_mu'].update(vals['smear_mu'])
		pltPars['smear_mu'] = float(vals['smear_mu'])
	if vals['smear_sig']:
		_VARS['window']['_smear_sig'].update(vals['smear_sig'])
		pltPars['smear_sig'] = float(vals['smear_sig'])


def main():

	#gui layout, layout is in rows, defined in order of list
	#Can add columns and other elements see guide

	#Column layout for plot parameters
	layout_plot = [
	[sg.T('Plot Parameters:',font='bold',size=(13,1)),sg.T('Current Values',size=(13,1),font='bold'),sg.T('Input Variables Below',font='bold')],
	[sg.Text('Number of Samples: ',size=(20,1)), sg.Text(size=(15,1),key='_nsamp',border_width=2,relief='raised'),sg.Input(key='nsamp')],
	[sg.Text('Plot xLow: ',size=(20,1)), sg.Text(size=(15,1),key='_low',border_width=2,relief='raised'),sg.Input(key='low')],
	[sg.Text('Plot xHigh: ',size=(20,1)), sg.Text(size=(15,1),key='_high',border_width=2,relief='raised'),sg.Input(key='high')],
	[sg.Text('Amplitude: ',size=(20,1)), sg.Text(size=(15,1),key='_amp',border_width=2,relief='raised'),sg.Input(key='amp')],
	[sg.Text('Phase (N*pi/2): ',size=(20,1)), sg.Text(size=(15,1),key='_phase',border_width=2,relief='raised'),sg.Input(key='phase')],
	[sg.Text('Offset: ',size=(20,1)), sg.Text(size=(15,1),key='_offset',border_width=2,relief='raised'),sg.Input(key='offset')],
	[sg.Text('Angular Freq(N*2*pi) : ',size=(20,1)), sg.Text(size=(15,1),key='_freq',border_width=2,relief='raised'),sg.Input(key='freq')],
	[sg.Text('Seed: ',size=(20,1)), sg.Text(size=(15,1),key='_sd',border_width=2,relief='raised'),sg.Input(key='sd')],
	[sg.Text('Smear: ',size=(20,1)), sg.Text(size=(15,1),key='_smear',border_width=2,relief='raised'),sg.Input(key='smear')],
	[sg.Text('Smear Mean: ',size=(20,1)), sg.Text(size=(15,1),key='_smear_mu',border_width=2,relief='raised'),sg.Input(key='smear_mu')],
	[sg.Text('Smear Sigma: ',size=(20,1)), sg.Text(size=(15,1),key='_smear_sig',border_width=2,relief='raised'),sg.Input(key='smear_sig')],
	]

	#Column layout for fit box
	layout_fit = [
	[sg.T('Test Fit Parameters:',font='bold')],
	[sg.Text('Goodness of fit (RSS): ',size=(20,1)),sg.Text(size=(15,1),key='_fit_gf',border_width=2,relief='raised')],
	[sg.Text('Amplitude: ',size=(20,1)), sg.Text(size=(15,1),key='_fit_amp',border_width=2,relief='raised'),sg.Input(key='fit_amp')],
	[sg.Text('Phase (N*pi/2): ',size=(20,1)), sg.Text(size=(15,1),key='_fit_phase',border_width=2,relief='raised'),sg.Input(key='fit_phase')],
	[sg.Text('Offset: ',size=(20,1)), sg.Text(size=(15,1),key='_fit_offset',border_width=2,relief='raised'),sg.Input(key='fit_offset')],
	[sg.Text('Angular Freq(N*2*pi) : ',size=(20,1)), sg.Text(size=(15,1),key='_fit_freq',border_width=2,relief='raised'),sg.Input(key='fit_freq')],
	]

	#Main GUI layout
	layout = [
	[sg.T('Figure: Sine Wave: F(t) = A*sin(2*pi*w*t - pi/2*phase)+ Offset',font='Helvetica 20 bold')],
	[sg.Canvas(key='figCanvas')],
	[sg.T('Plot Controls:',font='bold')],
	[sg.Canvas(key='controls_cv')],
	[sg.Canvas(key='fig_cv',size = (400*2*400))],
	[sg.Column(layout_plot),sg.Column(layout_fit)],
	[sg.Button('Update Plot',font=AppFont),sg.Button('Try Fit',font=AppFont),sg.Push(),sg.Button('Exit',font=AppFont)]
	]#End layout

	#Create the actual GUI object and store it globally
	_VARS['window'] = sg.Window('Test Plotting GUI',layout,finalize=True,resizable=True,element_justification="left",size=(1000,1200))

	#Make starting plot
	makePlot()
	
	#Show starting parameters
	_VARS['window']['_nsamp'].update(pltPars['nSamples'])
	_VARS['window']['_low'].update(pltPars['low'])
	_VARS['window']['_high'].update(pltPars['high'])
	_VARS['window']['_amp'].update(pltPars['amp'])
	_VARS['window']['_phase'].update(pltPars['phase'])
	_VARS['window']['_offset'].update(pltPars['offset'])
	_VARS['window']['_freq'].update(pltPars['freq'])
	_VARS['window']['_sd'].update(pltPars['sd'])
	_VARS['window']['_smear'].update(pltPars['smear'])
	_VARS['window']['_smear_mu'].update(pltPars['smear_mu'])
	_VARS['window']['_smear_sig'].update(pltPars['smear_sig'])
	#Starting Fit parameters
	_VARS['window']['_fit_amp'].update(fitVars['amp'])
	_VARS['window']['_fit_phase'].update(fitVars['phase'])
	_VARS['window']['_fit_offset'].update(fitVars['offset'])
	_VARS['window']['_fit_freq'].update(fitVars['freq'])



	# Display and interact with the Window using an Event Loop
	while True:
		event, values = _VARS['window'].read(timeout=200)
		if event == sg.WIN_CLOSED or event == 'Exit':
			break
		#Display updated values
		#Only update plot if requested
		if event == 'Update Plot':
			updateVars(values)
			updatePlot()
		if event == 'Try Fit':
			updateFitVars(values)
			updatePlotWithFit()

	_VARS['window'].close()


if __name__ == "__main__":
	main()




import numpy as np
import importlib.util

#Change the version name or path if this path is not correct: C:\\Program Files\\Lumerical\\v242\\api\\python\\lumapi.py
spec_win = importlib.util.spec_from_file_location('lumapi', 'C:\\Program Files\\Lumerical\\v232\\api\\python\\lumapi.py')

#Functions that perform the actual loading
lumapi = importlib.util.module_from_spec(spec_win) #windows
spec_win.loader.exec_module(lumapi)
print("Lumerical interface import done!")
print("You may get a warning, it is fine.")
It=lumapi.INTERCONNECT("Construct Matrix.icp")


# all length units 

lambda_0 = 1552.52438114966 * 1e-9 #nm

def phase_to_length (phase, lambda_0 = lambda_0):
    delta_L = phase / (2*np.pi) * lambda_0
    return delta_L


def forward(name_meter): 
    It.run()
    power = It.getresultdata(name_meter,"sum/power")
    It.switchtodesign()
    return power #return target powermeter output

def compute_gradient(name_dev, name_meter, step_phase):
    name_variable = "length" if name_dev[0] == "W" else "length 1"
    # calculate near the point of interest with plus minus 0.5 * step
    # First forward to calculate first data point
    It.select(name_dev)
    l0 = It.get(name_variable)
    It.select(name_dev)
    l0 -= phase_to_length(0.5 * step_phase)
    It.set(name_variable, l0)
    power_1 = forward(name_meter) * 100

    # second forward to calculate second data point
    It.select(name_dev)
    l0 += phase_to_length(step_phase)
    It.set(name_variable, l0)
    power_2 = forward(name_meter) * 100
    
    # Compute the gradient
    # print("current l0 (nm): ", l0 * 1e9)
    # print("previous power (dB): ", 10 * np.log10(power_1))
    # print("current power (dB): ", 10 * np.log10(power_2))

    # calculate the gradient from the initial step
    gradient = (power_2 - power_1) / step_phase
    
    return gradient, (power_1 + power_2) / 2



def minimize_power_gd(name_dev, name_meter, df,
                      step_phase=1e-4, gamma=2, precision=1e-3, max_iters=2500):
    """ minimize the power at name_meter by tuning name_dev
    step_phase: the perturbation of phase used to compute the gradient, should be small
    gamma: similar to learning rate, phase is updated according to this parameter
    precision: the cutoff for gradient
    max_iters: maximium number of runs
    """

    # Note now the initial step is phase step, because phase and power are more or less 
    # in the same dimension
    # all the optimization will be "delta_phase vs. power"
    
    name_variable = "length" if name_dev[0] == "W" else "length 1"
    iteration = 0
    # get the initial l0
    It.select(name_dev)
    l0 = It.get(name_variable)
    df=name_meter+df
    while iteration < max_iters:
        print()
        # calculate the gradient from the initial step
        gradient, power = compute_gradient(name_dev, name_meter, step_phase)
        
        # use gradient descent to update the phase
        d_phase = -gamma * gradient
        
        l0 += phase_to_length(d_phase)
        It.set(name_variable, l0)
        with open(df, "a") as file:
            file.write(f"{iteration}\t {gradient}\t {10 * np.log10(power)}\t{d_phase}\t{1e9 * phase_to_length(d_phase)}\t{l0 * 1e9} \n")
        print(f"iteration: {iteration}:\n\
              gradient: {gradient}, \tpower (dB): {10 * np.log10(power)}, \n\
              d_phase: {d_phase}, \tdl (nm): {1e9 * phase_to_length(d_phase)}, \tl0(nm): {l0 * 1e9}")
        
        # If the change in B is below the precision threshold, we're done
        if abs(gradient) < precision: 
            print("Phase Tunning stage complete")
            break
        
        iteration += 1
    
    if iteration == max_iters:
        print("Warning: maximium iterations achieved without small enough gradient") 
    
    return gradient, power



def pair_minimize(name_phase, name_refle,  name_meter, df, threshold=1e-3):
    """ Tuning the phase and reflectivity to minimize the output at power meter
    name_phase: name of the componenet that tunes the phase
    name_refle: name of the component that tunes the reflectivity (it is coupled with phase)
    name_meter: name of the power meter
    
    The parameters of gradient descents can be changed in minimize_power_gd
    """
    index = 0
    minimize_power_gd(name_phase, name_meter, df)
    _, power = minimize_power_gd(name_refle, name_meter, df)

    while index < 9: # max for total 10 iteration
        index +=1
        prev_power = power
        minimize_power_gd(name_phase, name_meter, df)
        _, power = minimize_power_gd(name_refle, name_meter, df)
        if abs(power-prev_power)/power < threshold: # changed here, 
            break
    if index == 9:
        print("Warning: maximium iterations achieved without convergence")



# Below code to two orthogonal input
######################################################
#input_phase
#input_phase2
#####################################################

def set_laser(power, phase):
    for i in range(len(power)): #set up laser power and phase using input 1 and phaseinput 1
        It.select("CWL_"+str(i+1))
        It.set("power", power[i]/sum(power))    # power of laser in W but doesn't matter, we normalize the sum to 1
        It.set("phase", phase[i])           # phase of laser
def Report(power, phase):
    set_laser(power, phase)
    It.run()
    with open("tunning_log.txt", "a") as file:
        Q4 = It.getresultdata("D21","sum/power") * 100
        Q3 = It.getresultdata("D22","sum/power") * 100
        Q2 = It.getresultdata("Output2","sum/power")
        Q1 = It.getresultdata("Output1","sum/power")

        file.write(f"Total power (dB): {10 * np.log10(sum(power))}\n")
        file.write(f"Q4 (dB): {10 * np.log10(Q4)}\n")
        file.write(f"Q3 (dB): {10 * np.log10(Q3)}\n")
        file.write(f"Q2 (dB): {10 * np.log10(Q2)}\n")
        file.write(f"Q1 (dB): {10 * np.log10(Q1)}\n")
    It.switchtodesign()
def tunning1 ():
    #with open("tunning process_df.txt", "w") as file:
        #file.write(f"iteration\tgradient\tpower (dB)\td_phase\tdl (nm)\tl0(nm)\n")
    pair_minimize("W14", "M13", "D13", "_df.txt")
    pair_minimize("W13", "M12", "D12", "_df.txt")
    pair_minimize("W12", "M11", "D11", "_df.txt")
    
    #Tunning complete/ report all power meter
    Report(power1,phase1)
    Report(power2,phase2)
    with open("tunning_log.txt", "a") as file:
        file.write(f"tunning 1 end\n")
    It.switchtodesign()

# Second algorithm for othogonal beam
def tunning2 ():
    pair_minimize("W23", "M22", "D22", "_df.txt")
    pair_minimize("W22", "M21", "D21", "_df.txt")
    
    #Tunning complete/ report all power meter
    Report(power1,phase1)
    Report(power2,phase2)
    with open("tunning_log.txt", "a") as file:
        file.write(f"tunning 2 end\n")
    It.switchtodesign()

'''power= [1,1,1,1] # in W
phase_1 = [0,0,0,0]
phase_2 = [0,np.pi,0,np.pi]'''
##### above simplified example


phase1 = np.array([-np.pi/2, np.pi/6, np.pi/4, np.pi/2])
phase2 = np.array([np.pi/2, np.pi/2, np.pi/4, np.pi/4])

power1 = np.array([2, 2, 1/4, 1])
power2 = np.array([7/4+np.sqrt(3), 1, 2, 3]) #tested to be orthogonal in Plot.ipynb
set_laser(power1, phase1)
tunning1()
set_laser(power2, phase2) #changed here
tunning2()



################################################
#No phase tunning
It.close()
It=lumapi.INTERCONNECT("Construct Matrix.icp")
set_laser(power1, phase1)
#tunning1
minimize_power_gd("M13", "D13", "_df_woPhase.txt")
minimize_power_gd("M12", "D12", "_df_woPhase.txt")
minimize_power_gd("M11", "D11", "_df_woPhase.txt")
Report(power1,phase1)
Report(power2,phase2)
with open("tunning_log.txt", "a") as file:
    file.write(f"tunning 1 end\n")
It.switchtodesign()

set_laser(power2, phase2)
#tunning2
minimize_power_gd("M22", "D22", "_df_woPhase.txt")
minimize_power_gd("M21", "D21", "_df_woPhase.txt")
Report(power1,phase1)
Report(power2,phase2)
with open("tunning_log.txt", "a") as file:
    file.write(f"tunning 2 end\n")
It.switchtodesign()
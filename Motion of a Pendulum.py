from cmath import pi, sqrt
import numpy as np
from numpy import cos, sin
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
    

#function to graph equation of motion 
def graph(t, solver, linear_eq_theory):
    """defines function to plot graph"""
    vel, non_linear, linear, time1, time2 = solver
    fig, ax = plt.subplots(1, 1)
    line2, = plt.plot(time2, non_linear, label="non-linear")
    #plot analytical solution and numerical solution
    line, = plt.plot(time1, linear, label="numerical solution", alpha=0.5)
    line3, = plt.plot(t, linear_eq_theory, 'r--', label="analytical theory", alpha=0.5)
    
    #plot kinetic and potential energy of system
    # line4, = plt.plot(time1, 0.5 * m * (pow(vel, 2)), label="kinetic energy")
    # line5, = plt.plot(time1, m * g * l * (1 - cos(linear)), label="potential energy")
    
    #plot the sum of the kinetic and potential energies
    line6, = plt.plot(time1, (0.5 * m * (pow(vel, 2))) + (m * g * l * (1 - cos(linear))), label="total energy")
        
    plt.title("Motion of Pendulum")
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.legend(loc=1)

    plt.show()
    
    
def solver(y0, t):
    """defines functions to solve the non-linear ODE"""
    non_linear_sol = integrate.solve_ivp(fun=non_linear_eq,
                                            t_span=(t0, tf),
                                            y0=y0,
                                            method = "RK45",
                                            t_eval=t)
    
    linear_eq_sol = integrate.solve_ivp(fun=linear_eq_theta,
                                           t_span=(t0, tf),
                                           y0=y0,
                                           method = "RK45",
                                           t_eval=t)
    
    result_linear, vel = linear_eq_sol.y
    result_non_linear, b = non_linear_sol.y
    time1 = linear_eq_sol.t
    time2 = non_linear_sol.t
    return (vel, result_non_linear, result_linear, time1, time2)


def non_linear_eq(t, y0):
    """defines ODE equation for pendulum"""
    theta, omega = y0
    dydt = np.array([omega, -(g/l) * sin(theta)])
    return dydt


def linear_eq_theta(t, y0):
    """defines ODE for pendulum using small angle approx"""
    theta, omega = y0
    dydt = np.array([omega, -(g/l) * theta])
    return dydt



def linear_eq_theory(theta, t):
    """defines the general solution for simple pendulum"""
    omega = sqrt(g / l)
    theta1 = theta * cos(omega * t)
    return theta1

def main():
    """initializes variables and constants
    and calls the graph() function"""
    global l, g, m, anis, t0, tf
    anis = 1 #degree of anisotropy, 1 by default
    omega = 0.0 #initial angular velocity
    init_angle = 5.0 #initial angle
    theta = np.radians(init_angle) #converts angle to radians
    g = 9.81 #accelaration due to gravity
    l = 1.0 #length of pendulum 
    m = 1.0 #mass of pendulum
    period = (2 * pi) * sqrt(l / g) #calculates period
    y0 = (theta, omega) #declares initial conditions
    t0 = 0.0 #intial time
    tf = 3600.0 #final time in seconds
    dt = 0.001 # time steps
    t = np.arange(t0, tf, dt) #time values
    graph(t, solver(y0, t), linear_eq_theory(theta, t)) #calls function

#runs code
if __name__ == "__main__":
    main() 

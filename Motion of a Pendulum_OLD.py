from cmath import pi, sqrt
import numpy as np
from numpy import cos, sin, tan
from scipy import integrate
import matplotlib.pyplot as plt
    

#function to graph equation of motion 
def graph(t, solver, linear_eq_theory):
    
    """defines function to plot graph"""
    
    vel_linear, vel_nonliner, non_linear, linear, time1, time2 = solver
    
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    
    #plot non-linear solution
    line1, = ax1.plot(time2, non_linear, label="non-linear", alpha=0.5, color="green")
    
    #plot analytical solution and numerical solution
    line2, = ax1.plot(time1, linear, label="numerical solution", alpha=0.5, color="blue")
    line3, = ax1.plot(t, linear_eq_theory, label="analytical theory", alpha=0.25, color="orange")
    
    #plot the sum of the kinetic and potential energies
    line4, = ax2.plot(time1, (0.5 * M * (pow(vel_linear, 2))) + (M * G * L * (1 - cos(linear))), label="liner mechanical energy", color="red")
    line5, = ax2.plot(time1, (0.5 * M * (pow(vel_nonliner, 2))) + (M * G * L * (1 - cos(non_linear))), label="non_linear mechanical energy", color="black")

    ax1.set_title("1d Pendulum")
    ax2.set_title("Total mechanical energies of both systems")
    
    ax1.set_ylabel("Position")
    ax2.set_ylabel("Position")
    ax2.set_xlabel("Time")
    
    plt.legend(loc="best")
    
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
    
    result_linear, vel_linear = linear_eq_sol.y
    result_non_linear, vel_nonlinear = non_linear_sol.y
    
    time1 = linear_eq_sol.t
    time2 = non_linear_sol.t
    
    return (vel_linear, vel_nonlinear, result_non_linear, result_linear, time1, time2)
 
def pendulum_3d(y, t):
    
    dthetadt, dphidt, theta, phi = y[0], y[1], y[2], y[3]
    
    ddthetadt = dphidt * cos(theta) * sin(theta) - (G/L) * sin(theta)
    ddphidt = -2 * dthetadt * dphidt / tan(theta)
    
    return np.array([ddthetadt, ddphidt, dthetadt, dphidt])


def runge_kutta(y, t, dt):
    
    k1 = pendulum_3d(y, t)
    k2 = pendulum_3d(y+0.5*k1*dt, t+0.5*dt)
    k3 = pendulum_3d(y+0.5*k2*dt, t+0.5*dt)
    k4 = pendulum_3d(y+k3*dt, t+dt)

    return dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

def non_linear_eq(t, y0):
    
    """defines ODE equation for pendulum"""
    theta, OMEGA = y0
    dydt = np.array([OMEGA, -(G / L) * sin(theta)])
    
    return dydt


def linear_eq_theta(t, y0):
    
    """defines ODE for pendulum using small angle approx"""
    theta, OMEGA = y0
    dydt = np.array([OMEGA, -(G / L) * theta])
    
    return dydt


def linear_eq_theory(theta, t):
    """defines the general solution for simple pendulum"""
    OMEGA = sqrt(G / L)
    theta1 = theta * cos(OMEGA * t)
    
    return theta1


def main():
    
    """initializes variables and constants
    and calls the graph() function"""
    
    global L, G, M, t0, tf
    OMEGA = 0 #initial angular velocity
    INIT_ANGLE = 5.0 #initial angle
    theta = np.radians(INIT_ANGLE) #converts angle to radians
    G = 9.81 #accelaration due to gravity
    L = 1.0 #length of pendulum 
    M = 1.0 #mass of pendulum
    
    y0 = (theta, OMEGA) #declares initial conditions
    
    t0 = 0.0 #intial time
    tf = 3600.0 #final time in seconds
    dt = 0.00001 # time steps
    t = np.arange(t0, tf, dt) #time values
    
        
    graph(t, solver(y0, t), linear_eq_theory(theta, t)) #calls function

#runs code
if __name__ == "__main__":
    main() 

from cmath import pi, sqrt
import numpy as np
from numpy import cos, sin, tan
import matplotlib.pyplot as plt
import time
    
    
def graph(time, theta1, theta2, theta3, theta4, x, y, vel1, vel2, vel3):
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True)
    fig1, ax4 = plt.subplots(1, 1)
    
    line1, = ax1.plot(time, theta2, label="Non Linear")
    line2, = ax1.plot(time, theta3, label="Linear")
    line3, = ax1.plot(time, theta4, label="analytical")
    
    line4, = ax2.plot(time, theta1, label="2d Pendulum")
    line5, = ax4.plot(x, y, label="x, y co-ords 2d Pendulum")
    
    line6, = ax3.plot(time, (0.5 * M * (pow(vel3, 2))) + (M * G * L * (1 - cos(theta3))), label="linear mechanical energy")
    line7, = ax3.plot(time, (0.5 * M * (pow(vel2, 2))) + (M * G * L * (1 - cos(theta2))), label="non_linear mechanical energy", color="black")
    
    ax1.set_title("1d Pendulum")
    ax2.set_title("2d Pendulum")
    ax3.set_title("Total mechanical energies")
    ax4.set_title("Focault Pendulum")
    
    ax1.set_ylabel("Position")
    ax2.set_ylabel("Position")
    ax4.set_ylabel("Y")
    
    ax3.set_xlabel("Time")
    ax4.set_xlabel("X")   
    
    ax1.legend(loc="best")
    ax2.legend(loc="best")
    ax3.legend(loc="best")
    ax4.legend(loc="best")
    
    plt.show()


def pendulum_2d(y, t):
    dthetadt, dphidt, theta, phi = y[0], y[1], y[2], y[3]

    ddthetadt = dphidt**2 * cos(theta) * sin(theta) - (G/L) * sin(theta)
    ddphidt = (-2 * dthetadt * dphidt) / tan(theta)
    
    result = np.array([ddthetadt, ddphidt, dthetadt, dphidt])
    
    return result


def non_linear_eq(y, t):
    dthetadt, theta = y[0], y[1]
    
    ddthetadt = -(G / L) * sin(theta)
    
    return np.array([ddthetadt, dthetadt])

def linear_eq_theta(y, t):
    dthetadt, theta = y[0], y[1]
    
    ddthetadt = -(G / L) * theta
    
    return np.array([ddthetadt, dthetadt])

def linear_eq_theory(t, theta0):
    omega = np.real(sqrt(G / L))
    result = theta0 * cos(omega * t)
    
    return result


def runge_kutta_2dPendulum(y, t, dt):
    """Runge kutta method for 2d pendulum"""
    k1 = pendulum_2d(y, t)
    k2 = pendulum_2d(y+0.5*k1*dt, t+0.5*dt)
    k3 = pendulum_2d(y+0.5*k2*dt, t+0.5*dt)
    k4 = pendulum_2d(y+k3*dt, t+dt)

    result = dt * (k1 + (2 * k2) + (2 * k3) + k4) / 6  

    return result

def runge_kutta_1dPendulum_nonlinear(y, t, dt):
    """Runge kutta method for non linear solution"""
    k1 = non_linear_eq(y, t)
    k2 = non_linear_eq(y+0.5*k1*dt, t+0.5*dt)
    k3 = non_linear_eq(y+0.5*k2*dt, t+0.5*dt)
    k4 = non_linear_eq(y+k3*dt, t+dt)

    return dt * (k1 + (2 * k2) + (2 * k3) + k4) / 6


def runge_kutta_1dPendulum_linear(y, t, dt):
    """Runge kutta method for linear solution"""
    k1 = linear_eq_theta(y, t)
    k2 = linear_eq_theta(y+0.5*k1*dt, t+0.5*dt)
    k3 = linear_eq_theta(y+0.5*k2*dt, t+0.5*dt)
    k4 = linear_eq_theta(y+k3*dt, t+dt)

    return dt * (k1 + (2 * k2) + (2 * k3) + k4) / 6
    
    
def main():
    """Defines main function"""
    global G, L, M, OMEGA
    OMEGA = 0.000072921
    M = 1
    G = 9.81
    L = 1
    INIT_THETA_ANGLE = 45
    INIT_PHI_ANGLE = 45
    theta0 = np.radians(INIT_THETA_ANGLE)#initial angle
    phi0 = np.radians(INIT_PHI_ANGLE) #intial angle
    theta_vel0 = 0 #initial theta velocity
    phi_vel0 = 1 #initial phi velocity

    #time values
    t0 = 0 #initial
    tf = 100 #final
    dt = 0.025 #times step
    time = np.arange(t0, tf, dt) #array of time values

    #final values for spherical coordinates
    theta1 = np.array([]) #2d pendulum
    theta2 = np.array([]) #1d non linear
    theta3 = np.array([]) #1d linear
    theta4 = np.array([]) #1d analytical
    
    #final values for velocities
    vel1 = np.array([]) #2d pendulum
    vel2 = np.array([]) #1d non linear
    vel3 = np.array([]) #1d linear
    
    #inital conditions for motion of pendulum
    y0 = np.array([theta_vel0, phi_vel0, theta0, phi0]) #2d pendulum 
    y1 = np.array([theta_vel0, theta0]) #1d non linear [linear_velocity, theta]
    y2 = np.array([theta_vel0, theta0]) #1d linear [linear_velocity, theta]

    #final x and y co-ordinates
    final_xpos = np.array([])
    final_ypos = np.array([]) 

    #get values for 2d pendulum
    for t in time:
        y0 = y0 + runge_kutta_2dPendulum(y0, t, dt)
        
        theta1 = np.append(theta1, y0[2])
        vel1 = np.append(vel1, y0[0])
        
        x = L * sin(y0[2]) * cos(y0[3])
        y = L * sin(y0[2]) * sin(y0[3])
        
        final_xpos = np.append(final_xpos, x)
        final_ypos = np.append(final_ypos, y)
             
    #get values for 1d pendulum nonlinear
    for t in time:
        y1 = y1 + runge_kutta_1dPendulum_nonlinear(y1, t, dt)
        
        theta2 = np.append(theta2, y1[1])
        vel2 = np.append(vel2, y1[0])
        
    #get values for 1d pendulum linear
    for t in time:
        y2 = y2 + runge_kutta_1dPendulum_linear(y2, t, dt)
        
        theta3 = np.append(theta3, y2[1])
        vel3 = np.append(vel3, y2[0])
    
    #get values for linear analytical   
    for t in time:
        theta4 = np.append(theta4, linear_eq_theory(t+dt, theta0))
        
    #call graph function
    graph(time, theta1, theta2, theta3, theta4, final_xpos, final_ypos, vel1, vel2, vel3)
        
        
if __name__ == "__main__":
    main() 


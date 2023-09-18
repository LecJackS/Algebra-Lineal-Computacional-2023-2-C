import matplotlib.pyplot as plt
import numpy as np



def proyectarPts(T, wz, corrimiento):
    assert(T.shape == (2,2)) # chequeo de matriz 2x2
    assert(T.shape[1] == wz.shape[0]) # multiplicacion matricial valida   
    xy = None
    ############### Insert code here!! ######################  
    xy = np.matmul(T, wz) # o T @ wz
    xy += corrimiento
    ############### Insert code here!! ######################
    return xy


def pointsGrid(corners):
    # crear 10 lineas horizontales
    [w1, z1] = np.meshgrid(np.linspace(corners[0,0], corners[1,0], 46),
                        np.linspace(corners[0,1], corners[1,1], 10))

    [w2, z2] = np.meshgrid(np.linspace(corners[0,0], corners[1,0], 10),
                        np.linspace(corners[0,1], corners[1,1], 46))

    w = np.concatenate((w1.reshape(1,-1),w2.reshape(1,-1)),1)
    z = np.concatenate((z1.reshape(1,-1),z2.reshape(1,-1)),1)
    wz = np.concatenate((w,z))
                         
    return wz
          
def vistform(T, wz, corrimiento, titulo=''):
    # transformar los puntos de entrada usando T
    xy = proyectarPts(T, wz, corrimiento)
    if xy is None:
        print('No fue implementada correctamente la proyeccion de coordenadas')
        return
    
    # calcular los limites para ambos plots
    minlim = np.min(np.concatenate((wz, xy), 1), axis=1)
    maxlim = np.max(np.concatenate((wz, xy), 1), axis=1)

    bump = [np.max(((maxlim[0] - minlim[0]) * 0.05, 0.1)),
            np.max(((maxlim[1] - minlim[1]) * 0.05, 0.1))]
    limits = [[minlim[0]-bump[0], maxlim[0]+bump[0]],
               [minlim[1]-bump[1], maxlim[1]+bump[1]]]             
    
    fig, (ax1, ax2) = plt.subplots(1, 2)         
    fig.suptitle(titulo)
    grid_plot(ax1, wz, limits, 'w', 'z')    
    grid_plot(ax2, xy, limits, 'x', 'y')
    
def grid_plot(ax, ab, limits, a_label, b_label):
    ax.plot(ab[0,:], ab[1,:], '.', markersize=2)
    ax.set(aspect='equal',
           xlim=limits[0], ylim=limits[1],
           xlabel=a_label, ylabel=b_label)
    

def main():
    print('Ejecutar el programa')
    # generar el tipo de transformacion dando valores a la matriz T
    
    # Ejercicio 2
    title = 'Ej 2: Sheer'
    #T = np.array([[1., 0.4],
    #              [0, 1.]])
    
    # Ejercicio 3
    title = 'Ej 3: Rotación'
    #theta = np.pi/10
    #T = np.array([[np.sin(theta), np.cos(theta)],[np.cos(theta),-np.sin(theta)]])
    
    # Ejercicio 4
    title = 'Ej 4: Rotación + escalamiento (incompleto)'
    # theta = np.pi/10
    # a = 3.
    # T = a * np.array([[np.sin(theta), np.cos(theta)],
    #               [np.cos(theta),-np.sin(theta)]])
    
    # Ejercicio 5
    title = 'Ej 5: Rotación + corrimiento'
    theta = np.pi/20
    a = 2.
    
    T = a * np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
    
    corners = np.array([[0,0],[100,100]])
    wz = pointsGrid(corners)
    print(wz.shape)
    c_x = 50
    c_y = 100
    corrimiento = np.array([c_x * np.ones(wz.shape[1]), c_y * np.ones(wz.shape[1])])
    vistform(T, wz, corrimiento, title)
    plt.show()
    
if __name__ == "__main__":
    main()

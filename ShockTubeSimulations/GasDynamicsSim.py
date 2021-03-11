# This is a code for using StanShock to get driver insert profile

# Things that need to be inputed:

# Mixture, Mixture properties: dirver and driven section mixture compositions, X4 and X1, and the corresponding .xml file for cantera
# Thermal, Thermodynamic properties: T5, p5, p1, gamma1, gamma4, W4, W1
# Sim, Simulation conditions: discretization sizes (nXCoarse, nXFine), tFinal, tTest
# Geometry, Shock tube geometries: driver and driven section length and diameter

import sys; sys.path.append('../../')
from stanShock import dSFdx, stanShock, smoothingFunction
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import time
import cantera as ct
from scipy.optimize import newton

class ShockSim:
    def __init__(self, Mixture, Thermal, Sim, Geometry, plot = True, saveData = False):
        self.Mixture = Mixture
        self.Thermal = Thermal
        self.Sim = Sim
        self.Geometry = Geometry
        self.plot = plot
        self.saveData = saveData

    def XT_Diagram_p4p1T1(self):
        fontsize = 12
        T1, p1, p4= self.Thermal['T1'], self.Thermal['p1'], self.Thermal['p4']
        tFinal = self.Sim['tFinal']
        MachReduction = 1#0.985  # account for shock wave attenuation
        nXFine = self.Sim['nXFine']  # mesh resolution
        LDriver, LDriven = self.Geometry['LDriver'], self.Geometry['LDriven']
        DDriver, DDriven = self.Geometry['DDriver'], self.Geometry['DDriven']
        plt.close("all")
        mpl.rcParams['font.size'] = fontsize
        plt.rc('text', usetex=True)
        # setup geometry
        xLower = -LDriver
        xUpper = LDriven
        xShock = 0.0
        Delta = 10 * (xUpper - xLower) / float(nXFine)
        geometry = (nXFine, xLower, xUpper, xShock)
        DInner = lambda x: np.zeros_like(x)
        dDInnerdx = lambda x: np.zeros_like(x)

        def DOuter(x): return smoothingFunction(x, xShock, Delta, DDriver, DDriven)

        def dDOuterdx(x): return dSFdx(x, xShock, Delta, DDriver, DDriven)

        A = lambda x: np.pi / 4.0 * (DOuter(x) ** 2.0 - DInner(x) ** 2.0)
        dAdx = lambda x: np.pi / 2.0 * (DOuter(x) * dDOuterdx(x) - DInner(x) * dDInnerdx(x))
        dlnAdx = lambda x, t: dAdx(x) / A(x)

        # set up the gasses
        u1 = 0.0
        u4 = 0.0  # initially 0 velocity
        mech = self.Mixture['mechanism']
        gas1 = ct.Solution(mech)
        gas4 = ct.Solution(mech)
        T4 = T1  # assumed
        gas1.TPX = T1, p1, self.Mixture['X1']
        gas4.TPX = T4, p4, self.Mixture['X4']

        # set up boundary condition
        boundaryConditions = ['reflecting', 'reflecting']
        state1 = (gas1, u1)
        state4 = (gas4, u4)

        try:
            self.Sim['alpha']
        except:
            self.Sim['alpha'] = 1
            self.Sim['beta'] = 1
            self.Sim['D_mul'] = 1

        ss = stanShock(gas1, initializeRiemannProblem=(state4, state1, geometry),
                       boundaryConditions=boundaryConditions,
                       cfl=.9,
                       outputEvery=100,
                       includeBoundaryLayerTerms=self.Sim['BoundaryLayer'],
                       reacting=self.Sim['Reacting'],
                       includeDiffusion=self.Sim['Diffusion'],
                       Tw=T1,  # assume wall temperature is in thermal eq. with gas
                       alpha = self.Sim['alpha'],
                       beta = self.Sim['beta'],
                       D_mul = self.Sim['D_mul'],
                       DOuter=DOuter,
                       dlnAdx=dlnAdx)
        ss.addXTDiagram("p")
        ss.addXTDiagram("T")
        try:
            self.Sim['ProbeLoc']
        except:
            self.Sim['ProbeLoc'] = 0
        ss.addProbe(max(ss.x)-self.Sim['ProbeLoc'])
        t0 = time.clock()
        ss.advanceSimulation(tFinal)
        t1 = time.clock()
        print("The process took ", t1 - t0)
        pNoInsert = np.array(ss.probes[0].p)
        tNoInsert = np.array(ss.probes[0].t)
        YNoInsert = np.array(ss.probes[0].Y)
        uNoInsert = np.array(ss.probes[0].u)
        rNoInsert = np.array(ss.probes[0].r)
        sNoInsert = np.array(ss.probes[0].s)
        TNoInsert = np.array(ss.thermoTable.getTemperature(rNoInsert, pNoInsert, YNoInsert))
        TWall = ss.Tw
        if self.plot:
            ss.plotXTDiagram(ss.XTDiagrams["t"], limits=[min(TNoInsert), max(TNoInsert)], saveData=True)
            TMatrix = ss.XTDiagram_variableMatrix
            timeXT = ss.XTDiagram_T
            positionXT = ss.XTDiagram_X
            ss.plotXTDiagram(ss.XTDiagrams["p"], limits=[min(pNoInsert)/101325, max(pNoInsert)/101325], saveData=True)
            pMatrix = ss.XTDiagram_variableMatrix
        # plot
        if self.plot:
            plt.figure()
            plt.plot(tNoInsert / 1e-3, pNoInsert / 1e5, 'b')  # label="$\mathrm{No\ Insert}$")
            plt.xlabel("$t\ [\mathrm{ms}]$")
            plt.ylabel("$p\ [\mathrm{bar}]$")
            plt.tight_layout()
            plt.figure()
            plt.plot(tNoInsert / 1e-3, TNoInsert, 'r')  # label="$\mathrm{No\ Insert}$")
            plt.xlabel("$t\ [\mathrm{ms}]$")
            plt.ylabel("$T\ [\mathrm{K}]$")
            plt.tight_layout()
        self.tNoInsert = tNoInsert
        self.pNoInsert = pNoInsert
        self.TNoInsert = TNoInsert
        self.YNoInsert = YNoInsert
        self.uNoInsert = uNoInsert
        self.sNoInsert = sNoInsert
        self.Tw = TWall
        self.TMatrix = TMatrix
        self.pMatrix = pMatrix
        self.timeXT = timeXT
        self.positionXT = positionXT
        # save driver insert profiles and pressure traces
        if self.saveData:
            np.savetxt('timeNoInsert.csv', tNoInsert, delimiter=',')
            np.savetxt('pNoInsert.csv', pNoInsert, delimiter=',')
            np.savetxt('TNoInsert.csv', TNoInsert, delimiter=',')
            np.savetxt('YNoInsert.csv', YNoInsert, delimiter=',')
            np.savetxt('TMatrix.csv', TMatrix, delimiter=',')
            np.savetxt('pMatrix.csv', pMatrix, delimiter=',')
            np.savetxt('timeXT.csv', timeXT, delimiter=',')
            np.savetxt('positionXT.csv', positionXT, delimiter=',')

    def XT_Diagram_T5p5p1(self):
        # input thermodynamic and shock tube parameters
        fontsize = 12
        tFinal = self.Sim['tFinal']
        p5, p1 = self.Thermal['p5'], self.Thermal['p1']
        T5 = self.Thermal['T5']
        # define driver and driven gases
        mech = self.Mixture['mechanism']
        gas1 = ct.Solution(mech)
        gas4 = ct.Solution(mech)
        gas1.X = self.Mixture['X1']
        gas4.X = self.Mixture['X4']
        # calculate mean molecular weight of driver and driven mixtures
        W4 = gas4.mean_molecular_weight
        W1 = gas1.mean_molecular_weight
        #print([W4, W1])
        # calculate specific heat ratios
        g4 = gas4.cp/gas4.cv
        g1 = gas1.cp/gas1.cv
        #print([g4, g1])
        MachReduction = 1#0.985  # account for shock wave attenuation
        nXFine = self.Sim['nXFine']  # mesh resolution
        LDriver, LDriven = self.Geometry['LDriver'], self.Geometry['LDriven']
        DDriver, DDriven = self.Geometry['DDriver'], self.Geometry['DDriven']
        plt.close("all")
        mpl.rcParams['font.size'] = fontsize
        plt.rc('text', usetex=True)

        # setup geometry
        xLower = -LDriver
        xUpper = LDriven
        xShock = 0.0
        Delta = 10 * (xUpper - xLower) / float(nXFine)
        geometry=(nXFine,xLower,xUpper,xShock)
        DInner = lambda x: np.zeros_like(x)
        dDInnerdx = lambda x: np.zeros_like(x)

        def DOuter(x): return smoothingFunction(x, xShock, Delta, DDriver, DDriven)

        def dDOuterdx(x): return dSFdx(x, xShock, Delta, DDriver, DDriven)

        A = lambda x: np.pi / 4.0 * (DOuter(x) ** 2.0 - DInner(x) ** 2.0)
        dAdx = lambda x: np.pi / 2.0 * (DOuter(x) * dDOuterdx(x) - DInner(x) * dDInnerdx(x))
        dlnAdx = lambda x, t: dAdx(x) / A(x)

        # compute gas dynamics
        def res(Ms1):
            return p5 / p1 - ((2.0 * g1 * Ms1 ** 2.0 - (g1 - 1.0)) / (g1 + 1.0)) \
                   * ((-2.0 * (g1 - 1.0) + Ms1 ** 2.0 * (3.0 * g1 - 1.0)) / (2.0 + Ms1 ** 2.0 * (g1 - 1.0)))

        Ms1 = newton(res, 2.0)
        Ms1 *= MachReduction
        T5oT1 = (2.0 * (g1 - 1.0) * Ms1 ** 2.0 + 3.0 - g1) \
                * ((3.0 * g1 - 1.0) * Ms1 ** 2.0 - 2.0 * (g1 - 1.0)) \
                / ((g1 + 1.0) ** 2.0 * Ms1 ** 2.0)
        T1 = T5 / T5oT1
        a1oa4 = np.sqrt(W4 / W1)
        p4op1 = (1.0 + 2.0 * g1 / (g1 + 1.0) * (Ms1 ** 2.0 - 1.0)) \
                * (1.0 - (g4 - 1.0) / (g4 + 1.0) * a1oa4 * (Ms1 - 1.0 / Ms1)) ** (-2.0 * g4 / (g4 - 1.0))
        p4 = p1 * p4op1
        print(f'Calculated p1 = {p1:4.1f}Pa, p4 = {p4:4.1f}Pa, T1 = {T1-273:4.1f}C')
        # set up the gasses
        u1 = 0.0
        u4 = 0.0  # initially 0 velocity
        T4 = T1  # assumed
        gas1.TPX = T1, p1, self.Mixture['X1']
        gas4.TPX = T4, p4, self.Mixture['X4']

        # set up boundary condition
        boundaryConditions = ['reflecting', 'reflecting']
        state1 = (gas1, u1)
        state4 = (gas4, u4)

        try:
            self.Sim['alpha']
        except:
            self.Sim['alpha'] = 1
            self.Sim['beta'] = 1
            self.Sim['D_mul'] = 1

        ss = stanShock(gas1, initializeRiemannProblem=(state4, state1, geometry),
                       boundaryConditions=boundaryConditions,
                       cfl=.9,
                       outputEvery=100,
                       includeBoundaryLayerTerms=self.Sim['BoundaryLayer'],
                       reacting = self.Sim['Reacting'],
                       includeDiffusion=self.Sim['Diffusion'],
                       Tw=T1,  # assume wall temperature is in thermal eq. with gas
                       alpha=self.Sim['alpha'],
                       beta=self.Sim['beta'],
                       D_mul=self.Sim['D_mul'],
                       DOuter=DOuter,
                       dlnAdx=dlnAdx)
        ss.addXTDiagram("p")
        ss.addXTDiagram("T")
        try:
            self.Sim['ProbeLoc']
        except:
            self.Sim['ProbeLoc'] = 0
        ss.addProbe(max(ss.x)-self.Sim['ProbeLoc'])#10-0.005)#max(ss.x))  # end wall probe
        t0 = time.clock()
        ss.advanceSimulation(tFinal)
        t1 = time.clock()
        print("The process took ", t1 - t0)
        pNoInsert = np.array(ss.probes[0].p)
        tNoInsert = np.array(ss.probes[0].t)
        YNoInsert = np.array(ss.probes[0].Y)
        uNoInsert = np.array(ss.probes[0].u)
        rNoInsert = np.array(ss.probes[0].r)
        sNoInsert = np.array(ss.probes[0].s)
        TNoInsert = np.array(ss.thermoTable.getTemperature(rNoInsert, pNoInsert, YNoInsert))
        TWall = ss.Tw
        if self.plot:
            ss.plotXTDiagram(ss.XTDiagrams["t"], limits=[T1 - 500, T5 + 500], saveData=True)
            TMatrix = ss.XTDiagram_variableMatrix
            timeXT = ss.XTDiagram_T
            positionXT = ss.XTDiagram_X
            ss.plotXTDiagram(ss.XTDiagrams["p"], limits=[p1 / 1e5 - 0.1, p5 / 1e5 + 0.3], saveData=True)
            pMatrix = ss.XTDiagram_variableMatrix
        # plot
        if self.plot:
            plt.figure()
            plt.plot(tNoInsert / 1e-3, pNoInsert / 1e5, 'b') #label="$\mathrm{No\ Insert}$")
            plt.xlabel("$t\ [\mathrm{ms}]$")
            plt.ylabel("$p\ [\mathrm{bar}]$")
            plt.tight_layout()
            plt.figure()
            plt.plot(tNoInsert / 1e-3, TNoInsert, 'r') #label="$\mathrm{No\ Insert}$")
            plt.xlabel("$t\ [\mathrm{ms}]$")
            plt.ylabel("$T\ [\mathrm{K}]$")
            plt.tight_layout()
        self.tNoInsert = tNoInsert
        self.pNoInsert = pNoInsert
        self.TNoInsert = TNoInsert
        self.YNoInsert = YNoInsert
        self.uNoInsert = uNoInsert
        self.sNoInsert = sNoInsert
        self.Tw = TWall
        self.TMatrix = TMatrix
        self.pMatrix = pMatrix
        self.timeXT = timeXT
        self.positionXT = positionXT
        # save driver insert profiles and pressure traces
        if self.saveData:
            np.savetxt('timeNoInsert.csv', tNoInsert, delimiter=',')
            np.savetxt('pNoInsert.csv', pNoInsert, delimiter=',')
            np.savetxt('TNoInsert.csv', TNoInsert, delimiter=',')
            np.savetxt('YNoInsert.csv', YNoInsert, delimiter=',')
            np.savetxt('TMatrix.csv', TMatrix, delimiter=',')
            np.savetxt('pMatrix.csv', pMatrix, delimiter=',')
            np.savetxt('timeXT.csv', timeXT, delimiter=',')
            np.savetxt('positionXT.csv', positionXT, delimiter=',')

    def XT_Diagram(self):
        try:
            self.Thermal['T5']
            self.XT_Diagram_T5p5p1()
        except:
            self.XT_Diagram_p4p1T1()
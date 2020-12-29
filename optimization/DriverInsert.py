# This is a code for using StanShock to get driver insert profile

# Things that need to be inputed:

# Mixture, Mixture properties: dirver and driven section mixture compositions, X4 and X1, and the corresponding .xml file for cantera
# Thermal, Thermodynamic properties: T5, p5, p1, gamma1, gamma4, W4, W1
# Sim, Simulation conditions: discretization sizes (nXCoarse, nXFine), tFinal, tTest
# Geometry, Shock tube geometries: driver and driven section length and diameter

import sys; sys.path.append('../')
from stanShock import dSFdx, stanShock, smoothingFunction
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import time
import cantera as ct
from scipy.optimize import newton

class InsertOpt:
    def __init__(self, Mixture, Thermal, Sim, Geometry, plot = True, saveData = False):
        self.Mixture = Mixture
        self.Thermal = Thermal
        self.Sim = Sim
        self.Geometry = Geometry
        self.plot = plot
        self.saveData = saveData

    def GetInert(self):
        # input thermodynamic and shock tube parameters
        fontsize = 12
        tFinal = self.Sim['tFinal']
        p5, p1 = self.Thermal['p5'], self.Thermal['p1']
        T5 = self.Thermal['T5']
        g4, g1 = self.Thermal['g4'], self.Thermal['g1']
        W4, W1 = self.Thermal['W4'], self.Thermal['W1']
        MachReduction = 0.985  # account for shock wave attenuation
        nXCoarse, nXFine = self.Sim['nXCoarse'], self.Sim['nXFine']  # mesh resolution
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
        geometry = (nXCoarse, xLower, xUpper, xShock)
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

        # set up the gasses
        u1 = 0.0;
        u4 = 0.0;  # initially 0 velocity
        mech = self.Mixture['mixtureFile']
        gas1 = ct.Solution(mech)
        gas4 = ct.Solution(mech)
        T4 = T1;  # assumed
        gas1.TPX = T1, p1, self.Mixture['X1']
        gas4.TPX = T4, p4, self.Mixture['X4']

        # set up solver parameter
        boundaryConditions = ['reflecting', 'reflecting']
        state1 = (gas1, u1)
        state4 = (gas4, u4)
        ss = stanShock(gas1, initializeRiemannProblem=(state4, state1, geometry),
                       boundaryConditions=boundaryConditions,
                       cfl=.9,
                       outputEvery=100,
                       includeBoundaryLayerTerms=True,
                       Tw=T1,  # assume wall temperature is in thermal eq. with gas
                       DOuter=DOuter,
                       dlnAdx=dlnAdx)

        # solve
        t0 = time.clock()
        tTest = self.Sim['tTest']
        tradeoffParam = 1.0
        eps = 0.01 ** 2.0 + tradeoffParam * 0.01 ** 2.0
        ss.optimizeDriverInsert(tFinal, p5=p5, tTest=tTest, tradeoffParam=tradeoffParam, eps=eps, maxIter=100)
        t1 = time.clock()
        print("The process took ", t1 - t0)

        # recalculate at higher resolution with the insert
        geometry = (nXFine, xLower, xUpper, xShock)
        gas1.TPX = T1, p1, self.Mixture['X1']
        gas4.TPX = T4, p4, self.Mixture['X4']
        ss = stanShock(gas1, initializeRiemannProblem=(state4, state1, geometry),
                       boundaryConditions=boundaryConditions,
                       cfl=.9,
                       outputEvery=100,
                       includeBoundaryLayerTerms=True,
                       Tw=T1,  # assume wall temperature is in thermal eq. with gas
                       DOuter=DOuter,
                       DInner=ss.DInner,
                       dlnAdx=ss.dlnAdx)
        ss.addXTDiagram("p")
        ss.addXTDiagram("T")
        ss.addProbe(max(ss.x))  # end wall probe
        t0 = time.clock()
        ss.advanceSimulation(tFinal)
        t1 = time.clock()
        print("The process took ", t1 - t0)
        pInsert = np.array(ss.probes[0].p)
        tInsert = np.array(ss.probes[0].t)

        if self.plot:
            ss.plotXTDiagram(ss.XTDiagrams["t"], limits=[T1-100, T5+100])
            ss.plotXTDiagram(ss.XTDiagrams["p"], limits=[p1/1e5 - 0.5, p5/1e5 + 1])
        xInsert = ss.x
        DOuterInsert = ss.DOuter(ss.x)
        DInnerInsert = ss.DInner(ss.x)

        # setup geometry of discrete insert
        DIn = ss.DInner(ss.x)
        xIn = ss.x
        x_step = self.Sim['xStep']
        disX = xIn[0:-1:x_step]
        disD = DIn[0:-1:x_step]
        delta = 1
        dx = xIn[1] - xIn[0]

        def DInner_discrete(x):
            DInner_dis = np.zeros(x.shape)
            cnt = 0
            for X in x:
                if np.sum(X > np.array(disX)) < len(disD):
                    DInner_dis[cnt] = disD[np.sum(X > np.array(disX))]
                cnt = cnt + 1
            for d in range(0, int(np.floor(len(x) / x_step)) - 1):
                LowBond = int(d * x_step - delta)
                UpBond = int(d * x_step + delta)
                x_loc = x[LowBond:UpBond]
                DInner_dis[LowBond:UpBond] = smoothingFunction(x_loc, disX[d], 2 * delta * dx, disD[d], disD[d + 1])
            return DInner_dis

        # plt.plot(xIn, DIn)#
        # plt.plot(xIn, DInner_discrete(xIn), '.')#%%
        # plt.plot(xIn[0:-1:x_step], DIn[0:-1:x_step], 'r.')
        #plt.xlim((-2, 0))

        def dDInnerdx_dis(x):
            dDIndx = np.zeros(x.shape)
            for d in range(0, int(np.floor(len(x) / x_step)) - 1):
                LowBond = int(d * x_step - delta)
                UpBond = int(d * x_step + delta)
                x_loc = x[LowBond:UpBond]
                dDIndx[LowBond:UpBond] = dSFdx(x_loc, disX[d], 2 * delta * dx, disD[d], disD[d + 1])
            return dDIndx

        A_dis = lambda x: np.pi / 4.0 * (DOuter(x) ** 2.0 - DInner_discrete(x) ** 2.0)
        dAdx_dis = lambda x: np.pi / 2.0 * (DOuter(x) * dDOuterdx(x) - DInner_discrete(x) * dDInnerdx_dis(x))
        dlnAdx_dis = lambda x, t: dAdx_dis(x) / A(x)

        # recalculate at higher resolution with discrete insert
        geometry = (nXFine, xLower, xUpper, xShock)
        gas1.TPX = T1, p1, self.Mixture['X1']
        gas4.TPX = T4, p4, self.Mixture['X4']
        ss = stanShock(gas1, initializeRiemannProblem=(state4, state1, geometry),
                       boundaryConditions=boundaryConditions,
                       cfl=.9,
                       outputEvery=100,
                       includeBoundaryLayerTerms=True,
                       Tw=T1,  # assume wall temperature is in thermal eq. with gas
                       DOuter=DOuter,
                       DInner=DInner_discrete,
                       dlnAdx=dlnAdx_dis)
        ss.addXTDiagram("p")
        ss.addXTDiagram("T")
        ss.addProbe(max(ss.x))  # end wall probe
        t0 = time.clock()
        ss.advanceSimulation(tFinal)
        t1 = time.clock()
        print("The process took ", t1 - t0)
        pInsert_dis = np.array(ss.probes[0].p)
        tInsert_dis = np.array(ss.probes[0].t)

        if self.plot:
            ss.plotXTDiagram(ss.XTDiagrams["t"], limits=[T1-100, T5+100])
            ss.plotXTDiagram(ss.XTDiagrams["p"], limits=[p1/1e5 - 0.5, p5/1e5 + 1])
        xInsert_dis = ss.x
        DOuterInsert_dis = ss.DOuter(ss.x)
        DInnerInsert_dis = ss.DInner(ss.x)

        # recalculate at higher resolution without the insert
        gas1.TPX = T1, p1, self.Mixture['X1']
        gas4.TPX = T4, p4, self.Mixture['X4']
        ss = stanShock(gas1, initializeRiemannProblem=(state4, state1, geometry),
                       boundaryConditions=boundaryConditions,
                       cfl=.9,
                       outputEvery=100,
                       includeBoundaryLayerTerms=True,
                       Tw=T1,  # assume wall temperature is in thermal eq. with gas
                       DOuter=DOuter,
                       dlnAdx=dlnAdx)
        ss.addXTDiagram("p")
        ss.addXTDiagram("T")
        ss.addProbe(max(ss.x))  # end wall probe
        t0 = time.clock()
        ss.advanceSimulation(tFinal)
        t1 = time.clock()
        print("The process took ", t1 - t0)
        pNoInsert = np.array(ss.probes[0].p)
        tNoInsert = np.array(ss.probes[0].t)
        if self.plot:
            ss.plotXTDiagram(ss.XTDiagrams["t"], limits=[T1-100, T5+100])
            ss.plotXTDiagram(ss.XTDiagrams["p"], limits=[p1/1e5 - 0.5, p5/1e5 + 1])

        # plot
        if self.plot:
            plt.figure()
            plt.plot(tNoInsert / 1e-3, pNoInsert / 1e5, 'k', label="$\mathrm{No\ Insert}$")
            plt.plot(tInsert / 1e-3, pInsert / 1e5, 'r', label="$\mathrm{Optimized\ Insert}$")
            plt.plot(tInsert_dis / 1e-3, pInsert_dis / 1e5, '--b', label="$\mathrm{Optimized\ Discrete\ Insert}$")
            plt.xlabel("$t\ [\mathrm{ms}]$")
            plt.ylabel("$p\ [\mathrm{bar}]$")
            plt.legend(loc="best")
            plt.tight_layout()

            plt.figure()
            plt.axis('equal')
            plt.xlim((-2, 0.5))
            plt.plot(xInsert, DOuterInsert, 'k', label="$D_\mathrm{o}$")
            plt.plot(xInsert, DInnerInsert, 'r', label="$D_\mathrm{i}$")
            plt.plot(xInsert_dis, DInnerInsert_dis, 'b', label="$D_\mathrm{dis}$")
            plt.xlabel("$x\ [\mathrm{m}]$")
            plt.ylabel("$D\ [\mathrm{m}]$")
            plt.legend(loc="best")
            plt.tight_layout()

        self.tNoInsert = tNoInsert
        self.pNoInsert = pNoInsert
        self.tInsert = tInsert
        self.pInsert = pInsert
        self.tInsert_dis = tInsert_dis
        self.pInsert_dis = pInsert_dis

        self.xInsert = xInsert
        self.xInsert_dis = xInsert_dis
        self.DOuterInsert = DOuterInsert
        self.DInnerInsert = DInnerInsert
        self.DInnerInsert_dis = DInnerInsert_dis

        # save driver insert profiles and pressure traces
        if self.saveData:
            np.savetxt('tNoInsert.csv', tNoInsert, delimiter=',')
            np.savetxt('pNoInsert.csv', pNoInsert, delimiter=',')
            np.savetxt('tInsert.csv', tInsert, delimiter=',')
            np.savetxt('pInsert.csv', pInsert, delimiter=',')
            np.savetxt('tInsert_dis.csv', tInsert_dis, delimiter=',')
            np.savetxt('pInsert_dis.csv', pInsert_dis, delimiter=',')
            np.savetxt('xInsert.csv', xInsert, delimiter=',')
            np.savetxt('xInsert_dis.csv', xInsert_dis, delimiter=',')
            np.savetxt('DOuterInsert.csv', DOuterInsert, delimiter=',')
            np.savetxt('DInnerInsert.csv', DInnerInsert, delimiter=',')
            np.savetxt('DInnerInsert_dis.csv', DInnerInsert_dis, delimiter=',')

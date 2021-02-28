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
from scipy.interpolate import interp1d

class InsertOpt:
    def __init__(self, Mixture, Thermal, Sim, Geometry, plot = True, saveData = False):
        self.Mixture = Mixture
        self.Thermal = Thermal
        self.Sim = Sim
        self.Geometry = Geometry
        self.plot = plot
        self.saveData = saveData
    def GetInsert(self):
        # input thermodynamic and shock tube parameters
        fontsize = 12
        tFinal = self.Sim['tFinal']
        try:
            self.Thermal['p5']
            have_p5 = True
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
            # print([W4, W1])
            # calculate specific heat ratios
            g4 = gas4.cp / gas4.cv
            g1 = gas1.cp / gas1.cv
            MachReduction = 0.985  # account for shock wave attenuation

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
        except:
            # if T5, p5 and p1 are not defined, expect T1, p4 and p1 inputs
            T1 = self.Thermal['T1']
            p4 = self.Thermal['p4']
            p1 = self.Thermal['p1']
            mech = self.Mixture['mechanism']
            gas1 = ct.Solution(mech)
            gas4 = ct.Solution(mech)
            have_p5 = False

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

        # set up the gasses
        u1 = 0.0;
        u4 = 0.0;  # initially 0 velocity
        T4 = T1;  # assumed
        gas1.TPX = T1, p1, self.Mixture['X1']
        gas4.TPX = T4, p4, self.Mixture['X4']

        # set up solver parameter
        boundaryConditions = ['reflecting', 'reflecting']
        state1 = (gas1, u1)
        state4 = (gas4, u4)
        # setup conditions for compressibility correction
        try:
            # if no values of multipliers are defined, set the values to 1
            self.Sim['alpha']
        except:
            self.Sim['alpha'] = 1
            self.Sim['beta'] = 1
            self.Sim['D_mul'] = 1

        ss = stanShock(gas1, initializeRiemannProblem=(state4, state1, geometry),
                       boundaryConditions=boundaryConditions,
                       cfl=.9,
                       outputEvery=100,
                       includeBoundaryLayerTerms=True,
                       Tw=T1,  # assume wall temperature is in thermal eq. with gas
                       alpha=self.Sim['alpha'],
                       beta=self.Sim['beta'],
                       D_mul=self.Sim['D_mul'],
                       DOuter=DOuter,
                       dlnAdx=dlnAdx)

        # solve
        t0 = time.clock()
        tTest = self.Sim['tTest']
        try: self.Sim['tradeoff']
        except: self.Sim['tradeoff'] = 1
        tradeoffParam = self.Sim['tradeoff']
        eps = 0.01 ** 2.0 + tradeoffParam * 0.01 ** 2.0
        # if a p5 is provided, can optimize around the given p5 value
        if have_p5:
            ss.optimizeDriverInsert(tFinal, p5=p5, tTest=tTest, tradeoffParam=tradeoffParam, eps=eps, maxIter=100)
        else:
            ss.optimizeDriverInsert(tFinal, tTest=tTest, tradeoffParam=tradeoffParam, eps=eps, maxIter=100)
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
                       alpha=self.Sim['alpha'],
                       beta=self.Sim['beta'],
                       D_mul=self.Sim['D_mul'],
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
        rInsert = np.array(ss.probes[0].r)
        YInsert = np.array(ss.probes[0].Y)
        TInsert = np.array(ss.thermoTable.getTemperature(rInsert, pInsert, YInsert))

        if self.plot:
            ss.plotXTDiagram(ss.XTDiagrams["t"], limits=[min(TInsert), max(TInsert)])
            ss.plotXTDiagram(ss.XTDiagrams["p"], limits=[min(pInsert)/1e5, max(pInsert)/1e5])
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
                       alpha=self.Sim['alpha'],
                       beta=self.Sim['beta'],
                       D_mul=self.Sim['D_mul'],
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
        rInsert_dis = np.array(ss.probes[0].r)
        YInsert_dis = np.array(ss.probes[0].Y)
        TInsert_dis = np.array(ss.thermoTable.getTemperature(rInsert_dis, pInsert_dis, YInsert_dis))
        if self.plot:
            ss.plotXTDiagram(ss.XTDiagrams["t"], limits=[min(TInsert_dis), max(TInsert_dis)])
            ss.plotXTDiagram(ss.XTDiagrams["p"], limits=[min(pInsert_dis)/1e5, max(pInsert_dis)/1e5])
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
                       alpha=self.Sim['alpha'],
                       beta=self.Sim['beta'],
                       D_mul=self.Sim['D_mul'],
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
        rNoInsert = np.array(ss.probes[0].r)
        YNoInsert = np.array(ss.probes[0].Y)
        TNoInsert = np.array(ss.thermoTable.getTemperature(rNoInsert, pNoInsert, YNoInsert))
        if self.plot:
            ss.plotXTDiagram(ss.XTDiagrams["t"], limits=[min(TNoInsert), max(TNoInsert)])
            ss.plotXTDiagram(ss.XTDiagrams["p"], limits=[min(pNoInsert)/1e5, max(pNoInsert)/1e5])

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
            plt.plot([xShock, xShock], [-0.6, 0.6], 'k--')
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
    ####################################################################################################################
    def SimulateInsertContinuous(self, DOuterInsert, DInnerInsert, xInsert, x_step):
        '''
        Method to simulate the shock tube flow given a continuous driver insert profile
        '''

        # input thermodynamic and shock tube parameters
        fontsize = 12
        tFinal = self.Sim['tFinal']
        try:
            self.Thermal['p5']
            have_p5 = True
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
            # print([W4, W1])
            # calculate specific heat ratios
            g4 = gas4.cp / gas4.cv
            g1 = gas1.cp / gas1.cv
            MachReduction = 0.985  # account for shock wave attenuation

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
        except:
            # if T5, p5 and p1 are not defined, expect T1, p4 and p1 inputs
            T1 = self.Thermal['T1']
            p4 = self.Thermal['p4']
            p1 = self.Thermal['p1']
            mech = self.Mixture['mechanism']
            gas1 = ct.Solution(mech)
            gas4 = ct.Solution(mech)
            have_p5 = False

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
        geometry = (nXFine, xLower, xUpper, xShock)

        # ----------------------------------------------------------------------------------------
        # First simulate the continuous insert profile
        # ----------------------------------------------------------------------------------------
        # get geometry of area change from the inputted insert profile:

        # set the size of the input arrays to one dimensional for intepolation
        xInsert = xInsert.reshape(len(xInsert), )
        DOuterInsert = DOuterInsert.reshape(len(DOuterInsert), )
        DInnerInsert = DInnerInsert.reshape(len(DInnerInsert), )

        # Interpolate to get a function of the discrete insert profile
        dx = xInsert[1]-xInsert[0]
        xInsert = np.append(xInsert[0]-dx, xInsert)
        xInsert = np.append(xInsert, xInsert[-1]+dx)
        DOuterInsert = np.append(DOuterInsert[0], DOuterInsert)
        DOuterInsert = np.append(DOuterInsert, DOuterInsert[-1])
        DInnerInsert = np.append(DInnerInsert[0], DInnerInsert)
        DInnerInsert = np.append(DInnerInsert, DInnerInsert[-1])
        DOuter = interp1d(xInsert, DOuterInsert)
        # calculate the rate of outer diameter change with x
        def dDOuterdx(x):
            dDdx = np.zeros_like(x)
            i = 0
            for xloop in x:
                where = np.sum(xloop>xInsert)
                dDdx[i] =  (DOuterInsert[where]-DOuterInsert[where-1])/(xInsert[where]-xInsert[where-1])
                i = i+1
            return dDdx
        # Interpolate to get a function of the discrete insert profile
        DInner = interp1d(xInsert, DInnerInsert)
        # calculate the rate of inner diameter change with x
        def dDInnerdx(x):
            dDdx = np.zeros_like(x)
            i = 0
            for xloop in x:
                where = np.sum(xloop>xInsert)
                dDdx[i] =  (DInnerInsert[where]-DInnerInsert[where-1])/(xInsert[where]-xInsert[where-1])
                i = i+1
            return dDdx
        # calculate the area profile of the driver section
        A = lambda x: np.pi / 4.0 * (DOuter(x) ** 2.0 - DInner(x) ** 2.0)
        # calculate the rate of area change with x
        dAdx = lambda x: np.pi / 2.0 * (DOuter(x) * dDOuterdx(x) - DInner(x) * dDInnerdx(x))
        dlnAdx = lambda x, t: dAdx(x) / A(x)

        # set up the gasses
        u1 = 0.0
        u4 = 0.0  # initially 0 velocity
        T4 = T1  # assumed
        gas1.TPX = T1, p1, self.Mixture['X1']
        gas4.TPX = T4, p4, self.Mixture['X4']

        # set up solver parameter
        boundaryConditions = ['reflecting', 'reflecting']
        state1 = (gas1, u1)
        state4 = (gas4, u4)
        # setup conditions for compressibility correction
        try:
            # if no values of multipliers are defined, set the values to 1
            self.Sim['alpha']
        except:
            self.Sim['alpha'] = 1
            self.Sim['beta'] = 1
            self.Sim['D_mul'] = 1

        # recalculate at higher resolution with the insert
        gas1.TPX = T1, p1, self.Mixture['X1']
        gas4.TPX = T4, p4, self.Mixture['X4']
        ss = stanShock(gas1, initializeRiemannProblem=(state4, state1, geometry),
                       boundaryConditions=boundaryConditions,
                       cfl=.9,
                       outputEvery=100,
                       includeBoundaryLayerTerms=True,
                       Tw=T1,  # assume wall temperature is in thermal eq. with gas
                       alpha=self.Sim['alpha'],
                       beta=self.Sim['beta'],
                       D_mul=self.Sim['D_mul'],
                       DOuter=DOuter,
                       DInner=DInner,
                       dlnAdx=dlnAdx)
        ss.addXTDiagram("p")
        ss.addXTDiagram("T")
        ss.addProbe(max(ss.x))  # end wall probe
        t0 = time.clock()
        ss.advanceSimulation(tFinal)
        t1 = time.clock()
        print("The process took ", t1 - t0)
        pInsert = np.array(ss.probes[0].p)
        tInsert = np.array(ss.probes[0].t)
        rInsert = np.array(ss.probes[0].r)
        YInsert = np.array(ss.probes[0].Y)
        TInsert = np.array(ss.thermoTable.getTemperature(rInsert, pInsert, YInsert))

        if self.plot:
            ss.plotXTDiagram(ss.XTDiagrams["t"], limits=[min(TInsert), max(TInsert)])
            ss.plotXTDiagram(ss.XTDiagrams["p"], limits=[min(pInsert) / 1e5, max(pInsert) / 1e5])
        # ----------------------------------------------------------------------------------------
        # Now simulate discretized insert
        # ----------------------------------------------------------------------------------------
        # setup geometry of discrete insert

        DIn = ss.DInner(ss.x)
        xIn = ss.x
        disX = xIn[0:-1:x_step]
        disD = DIn[0:-1:x_step]
        delta = 1
        dx = xIn[1] - xIn[0]

        def DInner_discrete(x):
            DInner_dis = np.zeros_like(x)
            cnt = 0
            for X in x:
                if np.sum(X > np.array(disX)) < len(disD):
                    DInner_dis[cnt] = disD[np.sum(X > np.array(disX))]
                cnt = cnt + 1
            for d in range(0, int(np.floor(len(x) / x_step)) - 1):
                LowBond = int(d * x_step - delta)
                UpBond = int(d * x_step + delta)
                x_loc = x[LowBond:UpBond]
                try:
                    DInner_dis[LowBond:UpBond] = smoothingFunction(x_loc, disX[d], 2 * delta * dx, disD[d], disD[d + 1])
                except:
                    DInner_dis[LowBond:UpBond] = smoothingFunction(x_loc, disX[d], 2 * delta * dx, disD[d], 0)
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
                       alpha=self.Sim['alpha'],
                       beta=self.Sim['beta'],
                       D_mul=self.Sim['D_mul'],
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
        rInsert_dis = np.array(ss.probes[0].r)
        YInsert_dis = np.array(ss.probes[0].Y)
        TInsert_dis = np.array(ss.thermoTable.getTemperature(rInsert_dis, pInsert_dis, YInsert_dis))
        DInnerInsert_dis = ss.DInner(ss.x)
        xInsert_dis = ss.x
        if self.plot:
            ss.plotXTDiagram(ss.XTDiagrams["t"], limits=[min(TInsert_dis), max(TInsert_dis)])
            ss.plotXTDiagram(ss.XTDiagrams["p"], limits=[min(pInsert_dis)/1e5, max(pInsert_dis)/1e5])
        # ----------------------------------------------------------------------------------------
        # Now simulate without insert
        # ----------------------------------------------------------------------------------------
        # setup geometry of no insert tube
        DInner = lambda x: np.zeros_like(x)
        dDInnerdx = lambda x: np.zeros_like(x)

        def DOuter(x):
            return smoothingFunction(x, xShock, Delta, DDriver, DDriven)

        def dDOuterdx(x):
            return dSFdx(x, xShock, Delta, DDriver, DDriven)

        A = lambda x: np.pi / 4.0 * (DOuter(x) ** 2.0 - DInner(x) ** 2.0)
        dAdx = lambda x: np.pi / 2.0 * (DOuter(x) * dDOuterdx(x) - DInner(x) * dDInnerdx(x))
        dlnAdx = lambda x, t: dAdx(x) / A(x)

        gas1.TPX = T1, p1, self.Mixture['X1']
        gas4.TPX = T4, p4, self.Mixture['X4']
        ss = stanShock(gas1, initializeRiemannProblem=(state4, state1, geometry),
                       boundaryConditions=boundaryConditions,
                       cfl=.9,
                       outputEvery=100,
                       includeBoundaryLayerTerms=True,
                       Tw=T1,  # assume wall temperature is in thermal eq. with gas
                       alpha=self.Sim['alpha'],
                       beta=self.Sim['beta'],
                       D_mul=self.Sim['D_mul'],
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
        rNoInsert = np.array(ss.probes[0].r)
        YNoInsert = np.array(ss.probes[0].Y)
        TNoInsert = np.array(ss.thermoTable.getTemperature(rNoInsert, pNoInsert, YNoInsert))
        if self.plot:
            ss.plotXTDiagram(ss.XTDiagrams["t"], limits=[min(TNoInsert), max(TNoInsert)])
            ss.plotXTDiagram(ss.XTDiagrams["p"], limits=[min(pNoInsert) / 1e5, max(pNoInsert) / 1e5])

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
            plt.plot([xShock, xShock], [-0.8, 0.8], 'k--')
            plt.xlabel("$x\ [\mathrm{m}]$")
            plt.ylabel("$D\ [\mathrm{m}]$")
            plt.legend(loc="best")
            plt.tight_layout()
        if self.saveData:
            np.savetxt('tNoInsert.csv', tNoInsert, delimiter=',')
            np.savetxt('pNoInsert.csv', pNoInsert, delimiter=',')
            np.savetxt('tInsert.csv', tInsert, delimiter=',')
            np.savetxt('pInsert.csv', pInsert, delimiter=',')
            np.savetxt('tInsert_dis.csv', tInsert_dis, delimiter=',')
            np.savetxt('pInsert_dis.csv', pInsert_dis, delimiter=',')
    ####################################################################################################################
    def SimulateInsertDiscrete(self, disX, disD):
        '''
           Method to simulate the shock tube flow given a continuous driver insert profile
        '''

        # set up input thermodynamic and shock tube parameters
        fontsize = 12
        tFinal = self.Sim['tFinal']
        try:
            self.Thermal['p5']
            have_p5 = True
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
            # print([W4, W1])
            # calculate specific heat ratios
            g4 = gas4.cp / gas4.cv
            g1 = gas1.cp / gas1.cv
            MachReduction = 0.985  # account for shock wave attenuation

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
        except:
            # if T5, p5 and p1 are not defined, expect T1, p4 and p1 inputs
            T1 = self.Thermal['T1']
            p4 = self.Thermal['p4']
            p1 = self.Thermal['p1']
            mech = self.Mixture['mechanism']
            gas1 = ct.Solution(mech)
            gas4 = ct.Solution(mech)
            have_p5 = False

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

        # ----------------------------------------------------------------------------------------
        # Now simulate discretized insert
        # ----------------------------------------------------------------------------------------
        # set up geometry of discrete insert



        # define tube geometry
        def DOuter(x):
            return smoothingFunction(x, xShock, Delta, DDriver, DDriven)
        # define rate of change in tube geometry
        def dDOuterdx(x):
            return dSFdx(x, xShock, Delta, DDriver, DDriven)

        delta = 1 #setp size for smoothing function between discontinuities in diameter
        def DInner_discrete(x):
            # initialize insert diameter array
            DInner_dis = np.zeros_like(x)
            # get the step size between grid points
            dx = x[1] - x[0]
            # begin loop to calculate insert diameters
            cnt = 0
            for X in x:
                # write all diameter between each length interval to the same diameter of the step
                if np.sum(X > np.array(disX)) < len(disD):
                    DInner_dis[cnt] = disD[np.sum(X > np.array(disX))]
                cnt = cnt + 1
            # begin loop to insert smoothed profile between diameter steps
            for d in range(0, len(disX)):
                # find range of smoothing function
                LowBond = sum(disX[d] > x) - delta
                UpBond = sum(disX[d] > x) + delta
                x_loc = x[LowBond:UpBond]
                try:
                    DInner_dis[LowBond:UpBond] = smoothingFunction(x_loc, disX[d], 2 * delta * dx, disD[d], disD[d + 1])
                except: # when the loop forwards to the last insert, set the final diameter to zero
                    DInner_dis[LowBond:UpBond] = smoothingFunction(x_loc, disX[d], 2 * delta * dx, disD[d], 0)
            return DInner_dis

        # calculate the rate of change of insert diameter
        def dDInnerdx_dis(x):
            # initialize array
            dDIndx = np.zeros(x.shape)
            dx = x[1] - x[0] # calculate step size of gird
            for d in range(0, len(disX)):
                # find range of points previously smoothed, otherwise slope is zero
                LowBond = sum(disX[d]>x)-delta
                UpBond = sum(disX[d]>x)+delta
                x_loc = x[LowBond:UpBond]
                try:
                    dDIndx[LowBond:UpBond] = dSFdx(x_loc, disX[d], 2 * delta * dx, disD[d], disD[d + 1])
                except:# when the loop forwards to the last insert, set the final diameter to zero
                    dDIndx[LowBond:UpBond] = dSFdx(x_loc, disX[d], 2 * delta * dx, disD[d], 0)
            return dDIndx

        A_dis = lambda x: np.pi / 4.0 * (DOuter(x) ** 2.0 - DInner_discrete(x) ** 2.0)
        dAdx_dis = lambda x: np.pi / 2.0 * (DOuter(x) * dDOuterdx(x) - DInner_discrete(x) * dDInnerdx_dis(x))
        dlnAdx_dis = lambda x, t: dAdx_dis(x) / A_dis(x)

        # recalculate at higher resolution with discrete insert
        geometry = (nXFine, xLower, xUpper, xShock)
        # set up the gasses
        u1 = 0.0
        u4 = 0.0  # initially 0 velocity
        T4 = T1  # assumed

        # set up solver parameter
        boundaryConditions = ['reflecting', 'reflecting']
        state1 = (gas1, u1)
        state4 = (gas4, u4)
        # setup conditions for compressibility correction
        try:
            # test of values of alpha exist
            self.Sim['alpha']
        except:
            # if no values of multipliers are defined, set the values to 1
            self.Sim['alpha'] = 1
            self.Sim['beta'] = 1
            self.Sim['D_mul'] = 1

        # recalculate at higher resolution with the insert
        gas1.TPX = T1, p1, self.Mixture['X1']
        gas4.TPX = T4, p4, self.Mixture['X4']
        ss = stanShock(gas1, initializeRiemannProblem=(state4, state1, geometry),
                       boundaryConditions=boundaryConditions,
                       cfl=.9,
                       outputEvery=100,
                       includeBoundaryLayerTerms=True,
                       Tw=T1,  # assume wall temperature is in thermal eq. with gas
                       alpha=self.Sim['alpha'],
                       beta=self.Sim['beta'],
                       D_mul=self.Sim['D_mul'],
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
        rInsert_dis = np.array(ss.probes[0].r)
        YInsert_dis = np.array(ss.probes[0].Y)
        TInsert_dis = np.array(ss.thermoTable.getTemperature(rInsert_dis, pInsert_dis, YInsert_dis))
        DInnerInsert_dis = ss.DInner(ss.x)
        xInsert_dis = ss.x
        if self.plot:
            ss.plotXTDiagram(ss.XTDiagrams["t"], limits=[min(TInsert_dis), max(TInsert_dis)])
            ss.plotXTDiagram(ss.XTDiagrams["p"], limits=[min(pInsert_dis) / 1e5, max(pInsert_dis) / 1e5])
        # ----------------------------------------------------------------------------------------
        # Now simulate without insert
        # ----------------------------------------------------------------------------------------
        # setup geometry of no insert tube
        DInner = lambda x: np.zeros_like(x)
        dDInnerdx = lambda x: np.zeros_like(x)

        def DOuter(x):
            return smoothingFunction(x, xShock, Delta, DDriver, DDriven)

        def dDOuterdx(x):
            return dSFdx(x, xShock, Delta, DDriver, DDriven)

        A = lambda x: np.pi / 4.0 * (DOuter(x) ** 2.0 - DInner(x) ** 2.0)
        dAdx = lambda x: np.pi / 2.0 * (DOuter(x) * dDOuterdx(x) - DInner(x) * dDInnerdx(x))
        dlnAdx = lambda x, t: dAdx(x) / A(x)

        gas1.TPX = T1, p1, self.Mixture['X1']
        gas4.TPX = T4, p4, self.Mixture['X4']
        ss = stanShock(gas1, initializeRiemannProblem=(state4, state1, geometry),
                       boundaryConditions=boundaryConditions,
                       cfl=.9,
                       outputEvery=100,
                       includeBoundaryLayerTerms=True,
                       Tw=T1,  # assume wall temperature is in thermal eq. with gas
                       alpha=self.Sim['alpha'],
                       beta=self.Sim['beta'],
                       D_mul=self.Sim['D_mul'],
                       DOuter=DOuter,
                       dlnAdx=dlnAdx)
        ss.addXTDiagram("p")
        ss.addXTDiagram("T")
        ss.addProbe(max(ss.x))  # end wall probe
        t0 = time.clock()
        ss.advanceSimulation(tFinal)
        t1 = time.clock()
        print("The process took ", t1 - t0)
        # export property time history at the end wall
        pNoInsert = np.array(ss.probes[0].p)
        tNoInsert = np.array(ss.probes[0].t)
        rNoInsert = np.array(ss.probes[0].r)
        YNoInsert = np.array(ss.probes[0].Y)
        TNoInsert = np.array(ss.thermoTable.getTemperature(rNoInsert, pNoInsert, YNoInsert))
        # export driver insert profile
        xNoInsert = ss.x
        DOuterTube = ss.DOuter(ss.x)

        if self.plot:
            ss.plotXTDiagram(ss.XTDiagrams["t"], limits=[min(TNoInsert), max(TNoInsert)])
            ss.plotXTDiagram(ss.XTDiagrams["p"], limits=[min(pNoInsert) / 1e5, max(pNoInsert) / 1e5])

        # plot
        if self.plot:
            plt.figure()
            plt.plot(tNoInsert / 1e-3, pNoInsert / 1e5, 'k', label="$\mathrm{No\ Insert}$")
            plt.plot(tInsert_dis / 1e-3, pInsert_dis / 1e5, '--b', label="$\mathrm{With\ Discrete\ Insert}$")
            plt.xlabel("$t\ [\mathrm{ms}]$")
            plt.ylabel("$p\ [\mathrm{bar}]$")
            plt.legend(loc="best")
            plt.tight_layout()

            plt.figure()
            plt.axis('equal')
            plt.xlim((-2, 0.5))
            plt.plot(xNoInsert, DOuterTube, 'k', label="$D_\mathrm{o}$")
            plt.plot(xInsert_dis, DInnerInsert_dis, 'b', label="$D_\mathrm{dis}$")
            plt.plot([xShock, xShock], [-0.8, 0.8], 'k--')
            plt.xlabel("$x\ [\mathrm{m}]$")
            plt.ylabel("$D\ [\mathrm{m}]$")
            plt.legend(loc="best")
            plt.tight_layout()
        if self.saveData:
            np.savetxt('tNoInsert.csv', tNoInsert, delimiter=',')
            np.savetxt('pNoInsert.csv', pNoInsert, delimiter=',')
            np.savetxt('tInsert_dis.csv', tInsert_dis, delimiter=',')
            np.savetxt('pInsert_dis.csv', pInsert_dis, delimiter=',')
    #####################################################################################################################
    def SimulateInsert(self, discrete = None, **kwargs):
        '''
        Method to simulate shock tube gas dynamics given a driver insert prifile, can be discrete or continuous
        '''
        if discrete == None:
            raise Exception('Need to define if the input insert profile is discrete or continuous!')
        if discrete:
            inchToMeter = 0.0254
            # catalog of existing insert diameters
            KeyToDiameter = {
                'N':    4.5*inchToMeter,
                'M':    4*inchToMeter,
                'L+':   3.6875*inchToMeter,
                'L':    3.5625*inchToMeter,
                'K':    3.3125*inchToMeter,
                'J':    3.0625*inchToMeter,
                'I':    2.9375*inchToMeter,
                'H':    2.625*inchToMeter,
                'G':    2.5625*inchToMeter,
                'F+':   2.5*inchToMeter,
                'F':    2.375*inchToMeter,
                'E':    2.25*inchToMeter,
                'D':    2.125*inchToMeter,
                'D-':   2.0625*inchToMeter,
                'C+':   1.8125*inchToMeter,
                'C':    1.6875*inchToMeter,
                'B':    1.5*inchToMeter,
                'A':    1.4375*inchToMeter,
                'O':    1.125*inchToMeter
            }

            # export insert profile
            Insert = kwargs['Insert']
            # translate profile
            disD = np.zeros(len(Insert.items()))
            disX = np.zeros(len(Insert.items()))
            i = 0
            insertLength = 0
            for key, length in Insert.items():
                insertLength = insertLength + length
                disD[i] = KeyToDiameter[key]
                disX[i] = insertLength*inchToMeter
                i = i+1
            disX = disX-self.Geometry['LDriver']
            self.SimulateInsertDiscrete(disX, disD)
        else:
            DOuterInsert = kwargs['DOuterInsert']
            DInnerInsert = kwargs['DInnerInsert']
            xInsert = kwargs['xInsert']
            xStep = kwargs['xStep']

            self.SimulateInsertContinuous(DOuterInsert, DInnerInsert, xInsert, xStep)
package goglm

import (
	"fmt"
	"math"
	"strings"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func (glm *GLM) fitIRLS(start []float64, maxiter int) []float64 {

	dtol := 1e-8

	var linpred []float64
	var mn []float64
	var va []float64
	var lderiv []float64
	var irlsw []float64
	var adjy []float64
	var nparam mat.VecDense

	nvar := glm.NumParams()

	xty := make([]float64, nvar)
	xtx := make([]float64, nvar*nvar)

	var params []float64
	if start == nil {
		params = make([]float64, nvar)
	} else {
		params = start
	}

	var dev []float64

	for iter := 0; iter < maxiter; iter++ {
		glm.data.Reset()
		zero(xtx)
		zero(xty)
		var devi float64
		for glm.data.Next() {

			var yda, wgt, off []float64

			yda = glm.data.GetPos(glm.ypos).([]float64)
			n := len(yda)

			if glm.weightpos != -1 {
				wgt = glm.data.GetPos(glm.weightpos).([]float64)
			}
			if glm.offsetpos != -1 {
				off = glm.data.GetPos(glm.offsetpos).([]float64)
			}

			// Allocations
			linpred = resize(linpred, n)
			mn = resize(mn, n)
			va = resize(va, n)
			lderiv = resize(lderiv, n)
			irlsw = resize(irlsw, n)
			adjy = resize(adjy, n)

			zero(linpred)
			for j, k := range glm.xpos {
				xda := glm.data.GetPos(k).([]float64)
				for i, x := range xda {
					linpred[i] += params[j] * x / glm.xn[j]
				}
			}
			if off != nil {
				floats.AddTo(linpred, linpred, off)
			}

			if iter == 0 {
				glm.startingMu(yda, mn)
			} else {
				glm.link.InvLink(linpred, mn)
			}

			glm.link.Deriv(mn, lderiv)
			glm.vari.Var(mn, va)

			devi += glm.fam.Deviance(yda, mn, wgt, 1)

			// Create weights for WLS
			if wgt != nil {
				for i, _ := range yda {
					irlsw[i] = wgt[i] / (lderiv[i] * lderiv[i] * va[i])
				}
			} else {
				for i, _ := range yda {
					irlsw[i] = 1 / (lderiv[i] * lderiv[i] * va[i])
				}
			}

			// Create an adjusted response for WLS
			if off == nil {
				for i, _ := range yda {
					adjy[i] = linpred[i] + lderiv[i]*(yda[i]-mn[i])
				}
			} else {
				for i, _ := range yda {
					adjy[i] = linpred[i] + lderiv[i]*(yda[i]-mn[i]) - off[i]
				}
			}

			// Update the weighted moment matrices
			for j1, k1 := range glm.xpos {

				// Update x' w^-1 ya
				xda := glm.data.GetPos(k1).([]float64)
				for i, y := range adjy {
					xty[j1] += y * xda[i] * irlsw[i] / glm.xn[j1]
				}

				// Update x' w^-1 x
				for j2, k2 := range glm.xpos {
					xdb := glm.data.GetPos(k2).([]float64)
					for i, _ := range xda {
						xtx[j1*nvar+j2] += xda[i] * xdb[i] * irlsw[i] / (glm.xn[j1] * glm.xn[j2])
					}
				}
			}
		}

		// Update the parameters
		xtxm := mat.NewDense(nvar, nvar, xtx)
		xtyv := mat.NewVecDense(nvar, xty)
		err := nparam.SolveVec(xtxm, xtyv)
		if err != nil {
			for j := 0; j < nvar; j++ {
				fmt.Printf("%8d %12.4f %12.4f\n", j, xty[j], xtx[j*nvar+j])
			}
			panic(err)
		}
		params = nparam.RawVector().Data

		dev = append(dev, devi)
		if len(dev) > 3 && math.Abs(dev[len(dev)-1]-dev[len(dev)-2]) < dtol {
			break
		}
	}

	// Undo the scaling
	for j := range params {
		params[j] /= glm.xn[j]
	}

	return params
}

func (glm *GLM) startingMu(y []float64, mn []float64) {

	var q float64
	name := strings.ToLower(glm.fam.Name)
	if name == "binomial" {
		q = 0.5
	} else {
		q = floats.Sum(y) / float64(len(y))
	}
	for i, _ := range mn {
		mn[i] = (y[i] + q) / 2
		if mn[i] < 0.1 {
			mn[i] = 0.1
		}
	}
}

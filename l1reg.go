package goglm

import (
	"strings"

	"github.com/kshedden/dstream/dstream"
	"github.com/kshedden/statmodel"
)

// Adapter that satisfies statmodel.L1RegFitter
type glml1reg struct {
	GLM

	nobs  int
	nvar  int
	l1wgt []float64

	// If false, do not check if the update from the quadratic
	// approximation improves the objective function.
	checkStep bool
}

func (glm *glml1reg) LogLike(params []float64) float64 {
	nobs := float64(glm.nobs)
	return -glm.GLM.LogLike(params, 1) / nobs
}

func (glm *glml1reg) Score(params, score []float64) {
	glm.GLM.Score(params, 1, score)
	nobs := float64(glm.nobs)
	for j, _ := range params {
		score[j] = -score[j] / nobs
	}
}

func (glm *glml1reg) Hessian(params, hess []float64) {
	glm.GLM.Hessian(params, 1, statmodel.ExpHess, hess)
	nobs := float64(glm.nobs)
	for j, _ := range hess {
		hess[j] = -hess[j] / nobs
	}
}

func (glm *glml1reg) L1wgt() []float64 {
	return glm.l1wgt
}

func (glm *glml1reg) CheckStep() bool {
	return glm.checkStep
}

func (glm *glml1reg) Data() dstream.Reg {
	return glm.GLM.Data
}

func (glm *glml1reg) CloneWithNewData(newdata dstream.Reg) statmodel.L1RegFitter {
	newglm := glm
	newglm.GLM.Data = newdata
	return newglm
}

// f1tL1Reg fits an L1-regularized GLM.  The objective function to be
// minimized is -L(params)/n + l1wgt*|params|, where L() is the
// log-likelihood function calculated with the scale parameter set
// equal to 1.
func (glm *GLM) fitL1Reg() GLMResults {

	checkstep := true
	if strings.ToLower(glm.Fam.Name) == "gaussian" {
		checkstep = false
	}

	rglm := &glml1reg{
		GLM:       *glm,
		l1wgt:     glm.L1wgt,
		nobs:      glm.Data.NumObs(),
		nvar:      glm.Data.NumCov(),
		checkStep: checkstep,
	}

	params := statmodel.FitL1Reg(rglm)
	xnames := glm.IndRegModel.Data.XNames()

	scale := glm.EstimateScale(params)

	results := GLMResults{
		BaseResults: statmodel.NewBaseResults(glm, 0, params, xnames, nil),
		scale:       scale,
	}

	return results

}

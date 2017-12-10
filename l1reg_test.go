package goglm

/*
import (
	"testing"

	"github.com/kshedden/dstream/dstream"
)

type endx struct {
	family *Family
	data   dstream.Reg
	params []float64
	l1wgt  []float64
}

func TestEnet1(t *testing.T) {

	for _, ep := range []endx{
		{
			family: NewFamily("gaussian"),
			data:   data1(false),
			params: []float64{1.290837, -0.103586},
		},
		{
			family: NewFamily("poisson"),
			data:   data1(false),
			l1wgt:  []float64{0.1, 0.1, 0.1},
			params: []float64{0.115096, -0.069824},
		},
		{
			family: NewFamily("poisson"),
			data:   data1(true),
			l1wgt:  []float64{0.1, 0.1, 0.1},
			params: []float64{0.226147, -0.031570},
		},
		{
			family: NewFamily("binomial"),
			data:   data2(false),
			l1wgt:  []float64{0.05, 0.05, 0.05},
			params: []float64{-0.737198, 0.024176, 0.017089},
		},
		{
			family: NewFamily("binomial"),
			data:   data2(false),
			l1wgt:  []float64{0.1, 0.1, 0.1},
			params: []float64{-0.465363, 0, 0},
		},
		{
			family: NewFamily("binomial"),
			data:   data2(true),
			l1wgt:  []float64{0.1, 0.1, 0.1},
			params: []float64{-0.627042, 0.080486, 0.147098},
		},
	} {
		glm := NewGLM(ep.family, ep.data)
		glm.L1wgt = ep.l1wgt
		rslt := glm.Fit()
		if !vectorClose(rslt.Params(), ep.params, 1e-5) {
			t.Fail()
		}
	}
}
*/

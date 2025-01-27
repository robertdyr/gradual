//
// Created by robert on 1/25/25.
//

#pragma once
#include "tensor.hpp"

class GradientDescent {
   	public:
        float learning_rate;
		GradientDescent(float learning_rate);
	    void step(std::shared_ptr<Tensor> t);

	private:
		void do_step(std::shared_ptr<Tensor> t);
};

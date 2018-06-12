Tensor Properties
	
	Rank is the dimensionality of a tensor.

	Rank    Description   Example
	0	Scalar 	      x=90
	1	Vector	      v=[0,1,5,4,56,8]
	2	Matrix	      m=[[1,2,3],[7,8,9]]
	3 	3_tensor      c=[[[1,2,3],[7,8,9]],[[1,2,3],[7,8,9]],[[1,2,3],[7,8,9]]]

	Shape is what the data looks like in a tensor.

	Rank    Description   Example			Shape			
	0	Scalar 	      x=90			[]
	1	Vector	      v=[0,1,5,4,56,8]		[5]
	2	Matrix	      m=[[1,2,3],[7,8,9]]	[2,4]
	3 	3_tensor      c=[[[1,2,3],[7,8,9]],	[1,2,3]
				[[1,2,3],[7,8,9]],
				[[1,2,3],[7,8,9]]]


	Datatypes TF supports

	float32,float64
	int8,int16,int32,int64
	uint8,uint16
	string
	bool
	complex64,complex128
	qint8,qint16,quint8

	the qint,quint ar optimized to take less space in memory so when we use them the process of training could be 
	significally enhanced.

	Common methods used to obtain tensor properties
	
	get_shape() - returns shape
	reshape()   - changes shape
	rank	    - returns rank
	dtype	    - return data type
	cast 	    - change data type


	
	

program matrix2

	use sep

  	implicit none

  	integer              				:: ndata,nmodel,adj,imodel,idata
  	integer								:: build_m
  	character*256						:: matrix_name
	real,dimension(:,:),allocatable   	:: matrix
	real,dimension(:),allocatable		:: model,data
	

  	call sep_init(SOURCE)
	call from_param("build_m",build_m,0) !If 1 it will write the matrix to use for inversion
	call from_param("matrix_name",matrix_name,"tmp_matrix.H")
	
	
	! build lower triangular matrix
	if(build_m) then
		call from_param("nmodel",nmodel)	
		ndata=nmodel
		allocate(matrix(ndata,nmodel))
 		matrix = 0.0
 		!Making the matrix symmetric
 		do idata=1,ndata
			matrix(idata,idata) = -2.0
		end do
		do idata=1,ndata-1
			matrix(idata,idata+1) = 1.
			matrix(idata+1,idata) = 1.
		end do
		call sep_write(matrix,matrix_name)
		call to_history("n1",ndata,matrix_name)
		call to_history("n2",nmodel,matrix_name)
		stop !Exit the program (to apply need to be recalled)
	else
		call from_aux(matrix_name,"n1",ndata)
		call from_aux(matrix_name,"n2",nmodel)
		allocate(matrix(ndata,nmodel))
		call sep_read(matrix,matrix_name)
	endif
	
	call from_param("adj",adj)
	!Reading data or model vector	
	allocate(data(ndata));	data = 0.
	allocate(model(nmodel));model= 0.
	if (adj) call sep_read(data)
	if (.not. adj) call sep_read(model)
	
	if (adj) then
		! apply matrix (adj)
		do imodel=1,nmodel
			do idata=1,ndata
				model(imodel) = model(imodel) + matrix(idata,imodel) *  data(idata)
			end do
		end do	
		call sep_write(model)
		call to_history("n1",nmodel)
	else
		! apply matrix (fwd)
		do imodel=1,nmodel
			do idata=1,ndata
				data(idata) = data(idata) + matrix(idata,imodel) *  model(imodel)
			end do
		end do	
		call sep_write(data)
		call to_history("n1",ndata)
	end if
		
		
	deallocate(data,model,matrix)
	call sep_close()					
	
end program matrix2
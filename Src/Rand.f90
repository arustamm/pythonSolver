!!$
!!$=head1 NAME
!!$
!!$Rand - Fills Seplib file with random numbers 
!!$       (IMPORTANT overwrites input file)
!!$
!!$=head1 SYNOPSIS
!!$
!!$Rand  file1=in.H 
!!$
!!$=head1 INPUT PARAMETERS
!!$
!!$=over 4
!!$
!!$=item  file1-  sepfile
!!$
!!$       Seplib file to fill with random numbers
!!$
!!$=item  verb-  logical
!!$
!!$       [0] Verbosity
!!$
!!$=item  keep_seed-  logical
!!$
!!$       [0] Change random seed?
!!$
!!$=item  pct_print-  integer
!!$
!!$       [10] How often to print progress
!!$
!!$=back
!!$
!!$=head1 DESCRIPTION
!!$
!!$ Fills Seplib file with different random numbers
!!$
!!$=head1 CATEGORY
!!$
!!$B<util/vector>
!!$
!!$=cut
!!$>

program rand
use sep3d_struct_mod
use sep
implicit none
real    :: temp2(2)
integer, parameter:: dbl=selected_int_kind(15)
type(sep3d) :: file1
integer(kind=dbl) :: sz1,sz2
integer, parameter :: dblr=selected_real_kind(10,50)
complex(kind=dblr) :: dt1c
real (kind=dblr) :: dt1r,t1,t2
character(len=128) :: op,f1,f2
character(len=256) :: temp1,temp_ch
integer :: n1,n2,idim,maxsize,ez1,ez2,ip,ip2,i
logical :: two_file
integer :: ierr
complex,allocatable :: buf1c(:),buf2c(:),outc(:)
real,allocatable :: buf1r(:),buf2r(:),outr(:)
integer(kind=dbl) :: itemp,iprint,inext,block,done
integer :: blocks, srite, sreed, sseek
logical :: esize8,verb,keep_seed
real :: scr,sci

call sep_init(SOURCE)

call getch_add_string("head=/dev/null")
call auxinout("file1")
call set_alternate_putch("file1")
call getch_add_string("noheader=y")

call from_param("op",op,"rand")
call from_param("file1",f1)
call from_param("keep_seed",keep_seed,.false.)


call init_sep3d("file1",file1,"SCRATCH")
call from_param("maxsize",maxsize,100)
maxsize=maxsize*1024*1024/4
esize8=.false.
if(file1%data_format(1:1)=="C")then
   esize8=.true.
   maxsize=maxsize/2
end if
if(two_file) maxsize=maxsize/2

sz1=1
do i=1,size(file1%n)
  sz1=sz1*file1%n(i)
end do

if(esize8) then
  allocate(buf1c(maxsize),buf2c(maxsize),outc(maxsize))
else
  allocate(buf1r(maxsize),buf2r(maxsize),outr(maxsize))
end if
      
done=0
call from_param("verb",verb,.false.)
inext=sz1+1
if(verb) then
  call from_param("iprint",ip,10)
  iprint=real(ip)/100.*sz1
  inext=iprint
end if
sz1=sz1
do while(done < sz1)
  block=min(sz1-done,maxsize)
  blocks=block
  if(inext > done .and. inext  < done+block) then
    t1=sz1; t2=done; t2=t2/t1; ip=nint(t2*1000.)
    t1=sz1; t2=done+block; t2=t2/t1; ip2=nint(t2*1000.)
    write(0,*) "working from",ip/10," to ",ip2/10
    inext=inext+iprint
  end if
  if(esize8) then
      if(block*8/=sreed("file1",buf1c,blocks*8)) call seperr("trouble reading in file1")
  else
      if(block*4/=sreed("file1",buf1r,blocks*4))call seperr("trouble reading in file1")
  end if
  if(esize8) then
      do i=1,size(buf1c)
      	!change random number generator seed
		if(.not. keep_seed) call random_seed()
        call random_number(temp2)
        buf1c(i)=cmplx(temp2(1),temp2(2))
      end do
  else
  	 if(.not. keep_seed) call random_seed()
     call random_number(buf1r)
  end if 
  done=done+block
    if(esize8) then
      ierr= sseek("file1",-blocks*8,1)
      if(8*block/= srite("file1",buf1c,blocks*8)) call seperr("trouble writing outc file")
    else
      ierr= sseek("file1",-blocks*4,1)
      if(4*block/= srite("file1",buf1r,blocks*4)) call seperr("trouble writing outc file")
    end if

end do

call sep_close()
call exit (0)
end program

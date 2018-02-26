/*

 KEYWORDS limit

 SEE ALSO			Again

*/
/*
 * Keyword: limit 
 */

/*
 * Written by Ettore Biondi based on Clip 9/22/2016
 */
#include <sep3dc.h>
#include <sepmath.h>
#include <complex.h>
#define MY_SEP_DOC \
 sep_add_doc_line("NAME");\
 sep_add_doc_line("    Limit - Clip maximum or minimum value");\
 sep_add_doc_line("");\
 sep_add_doc_line("SYNOPSIS");\
 sep_add_doc_line("      Limit clip= chop=greater <  in.H > out.H");\
 sep_add_doc_line("");\
 sep_add_doc_line("INPUT PARAMETERS");\
 sep_add_doc_line("    clip - float");\
 sep_add_doc_line("               Set the clip value yourself");\
 sep_add_doc_line("");\
 sep_add_doc_line("    chop - char*");\
 sep_add_doc_line("               [no]:     values are not cliped.");\
 sep_add_doc_line("               less:     all values less than or equal to the clip value will be changed.");\
 sep_add_doc_line("               greater:  all values greater than the clip value will be changed.");\
 sep_add_doc_line("               to:       What the above values are changed to a value, or 'clip'.");\
 sep_add_doc_line("               clip:     if chop=g");\
 sep_add_doc_line("");\
 sep_add_doc_line("    n1,n2,n3,n4,n5,n6 - int");\
 sep_add_doc_line("              standard SEP cube dimensions.");\
 sep_add_doc_line("");\
 sep_add_doc_line("    axis - int");\
 sep_add_doc_line("              Sub cube to do individual clip/scale operations on. Defaults to");\
 sep_add_doc_line("              the entire dataset");\
 sep_add_doc_line("");\
 sep_add_doc_line("    max_size - int");\
 sep_add_doc_line("               [50] Maximum memory to use  in megabytes");\
 sep_add_doc_line("");\
 sep_add_doc_line("    j - int");\
 sep_add_doc_line("               [nsize/esize/max_size]  - Subsamples the dataset to calculate");\
 sep_add_doc_line("                clip values. For Scale option the entire dataset is used.");\
 sep_add_doc_line("");\
 sep_add_doc_line("DESCRIPTION");\
 sep_add_doc_line("      Limit - Impose maximum or minimum boundary values");\
 sep_add_doc_line("");\
 sep_add_doc_line("NOTES");\
 sep_add_doc_line("       Not working for complex data");\
 sep_add_doc_line("");\
 sep_add_doc_line("CATEGORY");\
 sep_add_doc_line("    seis/filter");\
 sep_add_doc_line("");
#include <sep.main>
/*----------------------------------------------------------------------------*/
MAIN ()
{
	int     n1, n2, n3, k, pos, esize ;
	float   clip, dclip, pclip, *x;
	char    retry[10], chop[10], to[20];
	float   to_value;
  char progname[256];
  char axis[9],j2;
  int maxsize,have,i,jskip;
  int verb,nouter,ndim,nblock,nproc,ndo,ibig,good,do_scale,do_clip;
  int provide,iouter,providec,provides,jlast,clip_pos,nreadi;
  sep3d input;
  double ninner,ndone,nread,nbig;
  complex *cbuf;
  float  *fbuf;
  float  *buf;
  float my_max,scale,mys;
  float puse,clipit;

  sep_begin_prog();
  init_3d();

  /*Initialize the dataset */
  if(0!=init_sep3d_tag("in",&input,"INPUT"))
    return(sepwarn(-1,"Trouble initializing tag from input \n"));

  /*verbosity level*/
  if(0==getch("verb","1",&verb)) verb=0;

  /*subcube do calculations on*/
  if(1==getch("axis","s",axis)) {
    ndim=MIN(input.ndims,atoi(&axis[(int)strlen(axis)-1]));
    if(ndim < 1 ) 
     seperr("illegal axis ending %d in axis=%s \n",ndim,axis);
  }
  else ndim=input.ndims;
  for(i=0,nbig=1; i < ndim; i++) nbig=nbig*input.n[i];
  for(i=ndim,nouter=1; i < input.ndims; i++) nouter=nouter*input.n[i];


  sep_prog(progname);
	if( !getch ("dclip", "f", &dclip) )  dclip = 1.;
	if( !fetch ("esize", "d", &esize) )  esize=4;
  if( !getch("max_size","d",&maxsize)) maxsize=50;
  maxsize=50*1000*1000/esize;


  providec=1;
	if( 0==getch ("clip", "f", &clip) ) providec=0;
  scale=1;
  provides=0; 
  if (1==providec)putch("clip","f",&clip);
   
  pclip = 100.;
  do_scale=0; do_clip=1;

  jskip=1;
  
  if(verb) fprintf(stderr,"subcube dim=%d size=%g nouter=%d jskip=%d\n",
      ndim,nbig,nouter,jskip);

  maxsize=nbig/jskip;

  strcpy( chop, "no"); 
	getch( "chop", "s", chop);


   /*this is disturbing*/

	if( chop[0]=='g') { strcpy( to, "clip"); }
	else if( chop[0]=='l') { strcpy( to, "clip"); }
	else {              strcpy( to, "0.0");  }
	getch( "to", "s", to);

	if( chop[0] != 'n') {
		putch( "chop", "s", chop);
		putch( "to",   "s", to);
	};

  if(providec ==1)  do_clip=1;

  if((providec ==0 && do_clip==1) ||(provides ==0 && do_scale==1)) provide=0; 
  else provide=1;


  fbuf=(float*) malloc(sizeof(float)*1000000);
  if(esize==8) cbuf=(complex*) malloc(sizeof(complex)*1000000);
  buf=(float*) malloc(sizeof(float)*maxsize);

  make_unpipe("in");

  for(iouter=0; iouter < nouter; iouter++){

  clipit=clip;
  if(verb) fprintf(stderr,"For section=%d clip value=%f \n",iouter,clipit);

   clipit*=dclip;
   


  if (to[0] == 'c') to_value = clipit;
  else sscanf (to, "%f", &to_value);
   if(clipit>=0.) clip_pos=1;
   else clip_pos=0;
  ndone=0;
  ibig=0;
  while(ndone < nbig ){
    nproc=(int)MIN(nbig-ndone,maxsize);
    if(esize==4) {
      nread=0;
      while(nread < nproc){
        ndo=MIN(1000000,nproc-nread);
        if(esize*ndo!=sreed("in",fbuf,esize*ndo))
           seperr("trouble reading in dataset \n");
        if(do_clip==1){
          for(k=0; k < ndo;k++)  {
	          if ( (chop[0] == 'l') &&  ((fbuf[k]) <= clipit)) {
    	  	  	fbuf[k] = to_value;
    		  }
    		  else if((chop[0] == 'g') &&  ((fbuf[k]) >= clipit)){
    		  	fbuf[k] = to_value;
    		  }  	  	
          }
        }
        if(esize*ndo!=srite("out",fbuf,esize*ndo))
           seperr("trouble reading in dataset \n");
        nread+=ndo; 
      }
    }
    else{
      seperr("Complex number not supported \n");
    }
    ndone+=nproc;
   }
 }
   
 sep_end_prog();
 return(0);
}

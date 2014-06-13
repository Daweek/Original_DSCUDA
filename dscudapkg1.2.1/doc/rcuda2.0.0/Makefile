TARGET        = userguide-j.pdf
FIGSJ         = system.eps softlayer.eps rcudacc.eps
MAINRDOC      = ../bin/rcudacc
RDOCS         = $(wildcard ../lib/*.rb)
RDOCTARGETDIR = ./rdocs

default:	$(TARGET)

all:	$(TARGET) rdocs

#
# pLaTeX generic
#

%.pdf:	%.dvi
	dvipdfmx $<

%.dvi:	%.tex
	platex $<;
	platex $<;
	platex $<;

#
# User's Guide
#
userguide-j.dvi:	userguide-j.tex $(FIGSJ)

#
# figures and photos
#
$(FIGSJ):	%.eps:	%.obj
	tgif -print -eps $<

#
# Ruby codes
#

rdocs:	$(MAINRDOC) $(RDOCS)
	rdoc -main $(MAINRDOC) $(RDOCS) --inline-source -o $(RDOCTARGETDIR) -a

rdocsd:	$(MAINRDOC) $(RDOCS)
	rdoc -main $(MAINRDOC) $(RDOCS) --inline-source -o $(RDOCTARGETDIR) -a -d

clean:
	-rm -rf *.aux *.log *.dvi *.toc *~ core $(RDOCTARGETDIR) \
	$(patsubst %.pdf,%.ps,$(TARGET))

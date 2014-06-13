TARGET        = userguide-j.pdf
# FIGSJ         = system.eps softlayer.eps dscudacpp.eps

default:	$(TARGET)

all:	$(TARGET)

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

# $(FIGSJ):	%.eps:	%.obj
# 	tgif -print -eps $<

clean:
	-rm -rf *.aux *.log *.dvi *.toc *~ core \
	$(patsubst %.pdf,%.ps,$(TARGET))

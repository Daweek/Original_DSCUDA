#ifndef DSCUDA_MACROS_H
#define DSCUDA_MACROS_H

#define WARN(lv, fmt, args...) if (lv <= dscudaWarnLevel()) fprintf(stderr, fmt, ## args);
#define WARNONCE(lv, fmt, args...) if (lv <= dscudaWarnLevel()) { \
        static int firstcall = 1;                                 \
        if (firstcall) {                                          \
            firstcall = 0;                                        \
            fprintf(stderr, fmt, ## args);                        \
        }                                                         \
    }

#define ALIGN_UP(off, align) (off) = ((off) + (align) - 1) & ~((align) - 1)
int dscudaWarnLevel(void);
void dscudaSetWarnLevel(int level);

#endif // DSCUDA_MACROS_H

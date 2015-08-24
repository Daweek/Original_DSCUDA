#ifndef DSCUDA_MACROS_H
#define DSCUDA_MACROS_H

#define WARN(lv, fmt, args...) if (lv <= dscudaWarnLevel()) fprintf(stderr, fmt, ## args);

#define WARNONCE(lv, fmt, args...) if (lv <= dscudaWarnLevel()) { \
        static int firstcall = 1;                      \
        if (firstcall) {                                          \
            firstcall = 0;                                        \
            fprintf(stderr, fmt, ## args);                        \
        }                                                         \
    }

int dscudaWarnLevel(void);
void dscudaSetWarnLevel(int level);

#endif // DSCUDA_MACROS_H

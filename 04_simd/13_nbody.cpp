#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N], j[N], tmp1[N], tmp2[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
    j[i] = i;
  }
  __m256 jvec = _mm256_load_ps(j);
  __m256 xvec = _mm256_load_ps(x);
  __m256 yvec = _mm256_load_ps(y);
  __m256 mvec = _mm256_load_ps(m);
  for(int i=0; i<N; i++) {
    __m256 ivec = _mm256_set1_ps(i);
    __m256 mask = _mm256_cmp_ps(ivec, jvec, _CMP_NEQ_OQ);
    __m256 rxvec = _mm256_sub_ps(_mm256_set1_ps(x[i]), xvec);
    __m256 ryvec = _mm256_sub_ps(_mm256_set1_ps(y[i]), yvec);
    __m256 rinvvec = _mm256_rsqrt_ps(_mm256_add_ps(_mm256_mul_ps(rxvec, rxvec), _mm256_mul_ps(ryvec,ryvec)));
    __m256 rsqinvvec = _mm256_mul_ps(rinvvec,rinvvec);
    __m256 rcubinvvec = _mm256_mul_ps(rinvvec,rsqinvvec);
    __m256 mperr3vec = _mm256_mul_ps(mvec,rcubinvvec);
    __m256 fxvec = _mm256_blendv_ps(_mm256_setzero_ps(),_mm256_mul_ps(rxvec,mperr3vec),mask);
    __m256 fyvec = _mm256_blendv_ps(_mm256_setzero_ps(),_mm256_mul_ps(ryvec,mperr3vec),mask);
    __m256 xtmp = _mm256_permute2f128_ps(fxvec,fxvec,1);
    __m256 ytmp = _mm256_permute2f128_ps(fyvec,fyvec,1);
    xtmp = _mm256_add_ps(fxvec,xtmp);
    xtmp = _mm256_hadd_ps(xtmp,xtmp);
    xtmp = _mm256_hadd_ps(xtmp,xtmp);
    _mm256_store_ps(tmp1,xtmp);
    ytmp = _mm256_add_ps(fyvec,ytmp);
    ytmp = _mm256_hadd_ps(ytmp,ytmp);
    ytmp = _mm256_hadd_ps(ytmp,ytmp);
    _mm256_store_ps(tmp2,ytmp);
    fx[i] -= tmp1[0];
    fy[i] -= tmp2[0];
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}

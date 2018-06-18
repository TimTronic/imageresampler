// resampler.cpp, Separable filtering image rescaler v2.21, Rich Geldreich - richgel99@gmail.com
// See unlicense at the bottom of resampler.h, or at http://unlicense.org/
//
// Feb. 1996: Creation, losely based on a heavily bugfixed version of Schumacher's resampler in Graphics Gems 3.
// Oct. 2000: Ported to C++, tweaks.
// May 2001: Continous to discrete mapping, box filter tweaks.
// March 9, 2002: Kaiser filter grabbed from Jonathan Blow's GD magazine mipmap sample code.
// Sept. 8, 2002: Comments cleaned up a bit.
// Dec. 31, 2008: v2.2: Bit more cleanup, released as public domain.
// June 4, 2012: v2.21: Switched to unlicense.org, integrated GCC fixes supplied by Peter Nagy <petern@crytek.com>, Anteru at anteru.net, and clay@coge.net,
// added Codeblocks project (for testing with MinGW and GCC), VS2008 static code analysis pass.
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <cassert>
#include <cstring>
#include "resampler.h"

#define resampler_assert assert

static inline int resampler_range_check(int v, int h) { (void)h; resampler_assert((v >= 0) && (v < h)); return v; }

#ifndef max
   #define max(a,b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
   #define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

#ifndef TRUE
   #define TRUE (1)
#endif

#ifndef FALSE
   #define FALSE (0)
#endif

#define RESAMPLER_DEBUG 0

#define M_PI 3.14159265358979323846

#define RR(value) Resampler::RR(value)

// Float to int cast with truncation.
static constexpr int cast_to_int(Resample_Real i)
{
   return (int)i;
}

// (x mod y) with special handling for negative x values.
static constexpr int posmod(int x, int y)
{
   if (x >= 0)
      return (x % y);
   else
   {
      int m = (-x) % y;

      if (m != 0)
         m = y - m;

      return (m);
   }
}

// To add your own filter, insert the new function below and update the filter table.
// There is no need to make the filter function particularly fast, because it's
// only called during initializing to create the X and Y axis contributor tables.

#define BOX_FILTER_SUPPORT RR(0.5)
static constexpr Resample_Real box_filter(Resample_Real t)    /* pulse/Fourier window */
{
   // make_clist() calls the filter function with t inverted (pos = left, neg = right)
   if ((t >= RR(-0.5)) && (t < RR(0.5)))
      return RR(1.0);
   else
      return RR(0.0);
}

#define TENT_FILTER_SUPPORT RR(1.0)
static constexpr Resample_Real tent_filter(Resample_Real t)   /* box (*) box, bilinear/triangle */
{
   if (t < RR(0.0))
      t = -t;

   if (t < RR(1.0))
      return RR(1.0) - t;
   else
      return RR(0.0);
}

#define BELL_SUPPORT RR(1.5)
static constexpr Resample_Real bell_filter(Resample_Real t)    /* box (*) box (*) box */
{
   if (t < RR(0.0))
      t = -t;

   if (t < RR(.5))
      return (RR(.75) - (t * t));

   if (t < RR(1.5))
   {
      t = (t - RR(1.5));
      return (RR(.5) * (t * t));
   }

   return RR(0.0);
}

#define B_SPLINE_SUPPORT RR(2.0)
static constexpr Resample_Real B_spline_filter(Resample_Real t)  /* box (*) box (*) box (*) box */
{
   if (t < RR(0.0))
      t = -t;

   if (t < RR(1.0))
   {
      Resample_Real tt = t * t;
      return ((RR(.5) * tt * t) - tt + RR(2.0 / 3.0));
   }
   else if (t < RR(2.0))
   {
      t = RR(2.0) - t;
      return (RR(1.0 / 6.0) * (t * t * t));
   }

   return RR(0.0);
}

// Dodgson, N., "Quadratic Interpolation for Image Resampling"
#define QUADRATIC_SUPPORT RR(1.5)
static constexpr Resample_Real quadratic(Resample_Real t, const Resample_Real R)
{
   if (t < RR(0.0))
      t = -t;
   if (t < QUADRATIC_SUPPORT)
   {
      Resample_Real tt = t * t;
      if (t <= RR(.5))
         return (RR(-2.0) * R) * tt + RR(.5) * (R + RR(1.0));
      else
         return (R * tt) + (RR(-2.0) * R - RR(.5)) * t + RR(3.0 / 4.0) * (R + RR(1.0));
   }
   else
      return RR(0.0);
}

static constexpr Resample_Real quadratic_interp_filter(Resample_Real t)
{
   return quadratic(t, RR(1.0));
}

static constexpr Resample_Real quadratic_approx_filter(Resample_Real t)
{
   return quadratic(t, RR(.5));
}

static constexpr Resample_Real quadratic_mix_filter(Resample_Real t)
{
   return quadratic(t, RR(.8));
}

// Mitchell, D. and A. Netravali, "Reconstruction Filters in Computer Graphics."
// Computer Graphics, Vol. 22, No. 4, pp. 221-228.
// (B, C)
// (1/3, 1/3)  - Defaults recommended by Mitchell and Netravali
// (1, 0)	   - Equivalent to the Cubic B-Spline
// (0, 0.5)		- Equivalent to the Catmull-Rom Spline
// (0, C)		- The family of Cardinal Cubic Splines
// (B, 0)		- Duff's tensioned B-Splines.
static constexpr Resample_Real mitchell(Resample_Real t, const Resample_Real B, const Resample_Real C)
{
   Resample_Real tt = t * t;

   if(t < RR(0.0))
      t = -t;

   if(t < RR(1.0))
   {
      t = (((RR(12.0) - RR(9.0) * B - RR(6.0) * C) * (t * tt))
         + ((RR(-18.0) + RR(12.0) * B + RR(6.0) * C) * tt)
         + (RR(6.0) - RR(2.0) * B));

      return t / RR(6.0);
   }
   else if (t < RR(2.0))
   {
      t = (((RR(-1.0) * B - RR(6.0) * C) * (t * tt))
         + ((RR(6.0) * B + RR(30.0) * C) * tt)
         + ((RR(-12.0) * B - RR(48.0) * C) * t)
         + (RR(8.0) * B + RR(24.0) * C));

      return t / RR(6.0);
   }

   return RR(0.0);
}

#define MITCHELL_SUPPORT RR(2.0)
constexpr static Resample_Real mitchell_filter(Resample_Real t)
{
   return mitchell(t, RR(1.0 / 3.0), RR(1.0 / 3.0));
}

#define CATMULL_ROM_SUPPORT RR(2.0)
constexpr static Resample_Real catmull_rom_filter(Resample_Real t)
{
   return mitchell(t, RR(0.0), RR(.5));
}

constexpr static Resample_Real sinc(Resample_Real x)
{
   x = (x * RR(M_PI));

   if ((x < RR(0.01)) && (x > RR(-0.01)))
      return RR(1.0) + x*x*(RR(-1.0/6.0) + x*x*RR(1.0/120.0));

   return std::sin(x) / x;
}

static Resample_Real clean(Resample_Real t)
{
   constexpr Resample_Real EPSILON = RR(.0000125);
   if (std::fabs(t) < EPSILON)
      return RR(0.0);
   return t;
}

//static double blackman_window(double x)
//{
//	return .42f + .50f * cos(M_PI*x) + .08f * cos(2.0f*M_PI*x);
//}

static Resample_Real blackman_exact_window(Resample_Real x)
{
   return RR(0.42659071) + RR(0.49656062) * std::cos(RR(M_PI)*x) + RR(0.07684867) * std::cos(RR(2.0 * M_PI)*x);
}

#define BLACKMAN_SUPPORT RR(3.0)
static Resample_Real blackman_filter(Resample_Real t)
{
   if (t < RR(0.0))
      t = -t;

   if (t < RR(3.0))
      //return clean(sinc(t) * blackman_window(t / 3.0f));
      return clean(sinc(t) * blackman_exact_window(t / RR(3.0)));
   else
      return RR(0.0);
}

#define GAUSSIAN_SUPPORT RR(1.25)
static Resample_Real gaussian_filter(Resample_Real t) // with blackman window
{
   if (t < 0)
      t = -t;
   if (t < GAUSSIAN_SUPPORT)
      return clean(std::exp(RR(-2.0) * t * t) * std::sqrt(RR(2.0 / M_PI)) * blackman_exact_window(t / GAUSSIAN_SUPPORT));
   else
      return RR(0.0);
}

// Windowed sinc -- see "Jimm Blinn's Corner: Dirty Pixels" pg. 26.
#define LANCZOS3_SUPPORT RR(3.0)
static Resample_Real lanczos3_filter(Resample_Real t)
{
   if (t < RR(0.0))
      t = -t;

   if (t < RR(3.0))
      return clean(sinc(t) * sinc(t / RR(3.0)));
   else
      return RR(0.0);
}

#define LANCZOS4_SUPPORT RR(4.0)
static Resample_Real lanczos4_filter(Resample_Real t)
{
   if (t < RR(0.0))
      t = -t;

   if (t < RR(4.0))
      return clean(sinc(t) * sinc(t / RR(4.0)));
   else
      return RR(0.0);
}

#define LANCZOS6_SUPPORT RR(6.0)
static Resample_Real lanczos6_filter(Resample_Real t)
{
   if (t < RR(0.0))
      t = -t;

   if (t < RR(6.0))
      return clean(sinc(t) * sinc(t / RR(6.0)));
   else
      return RR(0.0);
}

#define LANCZOS12_SUPPORT RR(12.0)
static Resample_Real lanczos12_filter(Resample_Real t)
{
   if (t < RR(0.0))
      t = -t;

   if (t < RR(12.0))
      return clean(sinc(t) * sinc(t / RR(12.0)));
   else
      return RR(0.0);
}

static constexpr Resample_Real bessel0(Resample_Real x)
{
   constexpr Resample_Real EPSILON_RATIO = RR(1E-16);

   Resample_Real xh = RR(0.5) * x;
   Resample_Real sum = RR(1.0);
   Resample_Real pow = RR(1.0);
   int k = 0;
   Resample_Real ds = RR(1.0);
   while (ds > sum * EPSILON_RATIO) // FIXME: Shouldn't this stop after X iterations for max. safety?
   {
      ++k;
      pow = pow * (xh / k);
      ds = pow * pow;
      sum = sum + ds;
   }

   return sum;
}

//static const Resample_Real KAISER_ALPHA = 4.0;
static Resample_Real kaiser(Resample_Real alpha, Resample_Real half_width, Resample_Real x)
{
   const Resample_Real ratio = (x / half_width);
   return bessel0(alpha * std::sqrt(Resample_Real(1) - ratio * ratio)) / bessel0(alpha);
}

#define KAISER_SUPPORT RR(3)
static Resample_Real kaiser_filter(Resample_Real t)
{
   if (t < RR(0.0))
      t = -t;

   if (t < KAISER_SUPPORT)
   {
      // db atten
      const Resample_Real att = RR(40.0);
      const Resample_Real alpha = (Resample_Real)(exp(log(RR(0.58417) * (att - RR(20.96))) * RR(0.4)) + RR(0.07886) * (att - RR(20.96)));
      //const Resample_Real alpha = KAISER_ALPHA;
      return clean(sinc(t) * kaiser(alpha, KAISER_SUPPORT, t));
   }

   return RR(0.0);
}

// filters[] is a list of all the available filter functions.
static struct
{
   char name[32];
   Resample_Real (*func)(Resample_Real t);
   Resample_Real support;
} g_filters[] =
{
   { "box",		            box_filter,			         BOX_FILTER_SUPPORT },
   { "tent",			      tent_filter,		         TENT_FILTER_SUPPORT },
   { "bell",			      bell_filter,	            BELL_SUPPORT },
   { "b-spline",	         B_spline_filter,	         B_SPLINE_SUPPORT },
   { "mitchell",	         mitchell_filter,	         MITCHELL_SUPPORT },
   { "lanczos3",	         lanczos3_filter,	         LANCZOS3_SUPPORT },
   { "blackman",	         blackman_filter,	         BLACKMAN_SUPPORT },
   { "lanczos4",	         lanczos4_filter,	         LANCZOS4_SUPPORT },
   { "lanczos6",	         lanczos6_filter,	         LANCZOS6_SUPPORT },
   { "lanczos12",          lanczos12_filter,          LANCZOS12_SUPPORT },
   { "kaiser",		         kaiser_filter,		         KAISER_SUPPORT },
   { "gaussian",	         gaussian_filter,	         GAUSSIAN_SUPPORT },
   { "catmullrom",         catmull_rom_filter,        CATMULL_ROM_SUPPORT },
   { "quadratic_interp",   quadratic_interp_filter,   QUADRATIC_SUPPORT },
   { "quadratic_approx",   quadratic_approx_filter,   QUADRATIC_SUPPORT },
   { "quadratic_mix",      quadratic_mix_filter,      QUADRATIC_SUPPORT },
};

static const int NUM_FILTERS = sizeof(g_filters) / sizeof(g_filters[0]);

/* Ensure that the contributing source sample is
* within bounds. If not, reflect, clamp, or wrap.
*/
int Resampler::reflect(const int j, const int src_x, const Boundary_Op boundary_op)
{
   int n;

   if (j < 0)
   {
      if (boundary_op == BOUNDARY_REFLECT)
      {
         n = -j;

         if (n >= src_x)
            n = src_x - 1;
      }
      else if (boundary_op == BOUNDARY_WRAP)
         n = posmod(j, src_x);
      else
         n = 0;
   }
   else if (j >= src_x)
   {
      if (boundary_op == BOUNDARY_REFLECT)
      {
         n = (src_x - j) + (src_x - 1);

         if (n < 0)
            n = 0;
      }
      else if (boundary_op == BOUNDARY_WRAP)
         n = posmod(j, src_x);
      else
         n = src_x - 1;
   }
   else
      n = j;

   return n;
}

// The make_clist() method generates, for all destination samples,
// the list of all source samples with non-zero weighted contributions.
Resampler::Contrib_List* Resampler::make_clist(
   int src_x, int dst_x, Boundary_Op boundary_op,
   Resample_Real (*Pfilter)(Resample_Real),
   Resample_Real filter_support,
   Resample_Real filter_scale,
   Resample_Real src_ofs)
{
   typedef struct
   {
      // The center of the range in DISCRETE coordinates (pixel center = 0.0f).
      Resample_Real center;
      int left, right;
   } Contrib_Bounds;

   int i, j, k, n, left, right;
   Resample_Real total_weight;
   Resample_Real xscale, center, half_width, weight;
   Contrib_List* Pcontrib;
   Contrib* Pcpool;
   Contrib* Pcpool_next;
   Contrib_Bounds* Pcontrib_bounds;

   if ((Pcontrib = (Contrib_List*)calloc(dst_x, sizeof(Contrib_List))) == NULL)
      return NULL;

   Pcontrib_bounds = (Contrib_Bounds*)calloc(dst_x, sizeof(Contrib_Bounds));
   if (!Pcontrib_bounds)
   {
      free(Pcontrib);
      return (NULL);
   }

   const Resample_Real oo_filter_scale = RR(1.0) / filter_scale;

   constexpr Resample_Real NUDGE = RR(0.5);
   xscale = dst_x / (Resample_Real)src_x;

   if (xscale < RR(1.0))
   {
      int total; (void)total;

      /* Handle case when there are fewer destination
      * samples than source samples (downsampling/minification).
      */

      // stretched half width of filter
      half_width = (filter_support / xscale) * filter_scale;

      // Find the range of source sample(s) that will contribute to each destination sample.

      for (i = 0, n = 0; i < dst_x; i++)
      {
         // Convert from discrete to continuous coordinates, scale, then convert back to discrete.
         center = ((Resample_Real)i + NUDGE) / xscale;
         center -= NUDGE;
         center += src_ofs;

         left   = cast_to_int(std::floor(center - half_width));
         right  = cast_to_int(std::ceil(center + half_width));

         Pcontrib_bounds[i].center = center;
         Pcontrib_bounds[i].left		= left;
         Pcontrib_bounds[i].right	= right;

         n += (right - left + 1);
      }

      /* Allocate memory for contributors. */

      if ((n == 0) || ((Pcpool = (Contrib*)calloc(n, sizeof(Contrib))) == NULL))
      {
         free(Pcontrib);
         free(Pcontrib_bounds);
         return NULL;
      }
      total = n;

      Pcpool_next = Pcpool;

      /* Create the list of source samples which
      * contribute to each destination sample.
      */

      for (i = 0; i < dst_x; i++)
      {
         int max_k = -1;
         Resample_Real max_w = RR(-1e+20);

         center = Pcontrib_bounds[i].center;
         left   = Pcontrib_bounds[i].left;
         right  = Pcontrib_bounds[i].right;

         Pcontrib[i].n = 0;
         Pcontrib[i].p = Pcpool_next;
         Pcpool_next += (right - left + 1);
         resampler_assert ((Pcpool_next - Pcpool) <= total);

         total_weight = 0;

         for (j = left; j <= right; j++)
            total_weight += (*Pfilter)((center - (Resample_Real)j) * xscale * oo_filter_scale);
         const Resample_Real norm = static_cast<Resample_Real>(RR(1.0) / total_weight);

         total_weight = 0;

#if RESAMPLER_DEBUG
         printf("%i: ", i);
#endif

         for (j = left; j <= right; j++)
         {
            weight = (*Pfilter)((center - (Resample_Real)j) * xscale * oo_filter_scale) * norm;
            if (weight == RR(0.0))
               continue;

            n = reflect(j, src_x, boundary_op);

#if RESAMPLER_DEBUG
            printf("%i(%f), ", n, weight);
#endif

            /* Increment the number of source
            * samples which contribute to the
            * current destination sample.
            */

            k = Pcontrib[i].n++;

            Pcontrib[i].p[k].pixel  = (unsigned short)(n);       /* store src sample number */
            Pcontrib[i].p[k].weight = weight; /* store src sample weight */

            total_weight += weight;          /* total weight of all contributors */

            if (weight > max_w)
            {
               max_w = weight;
               max_k = k;
            }
         }

#if RESAMPLER_DEBUG
         printf("\n\n");
#endif

         //resampler_assert(Pcontrib[i].n);
         //resampler_assert(max_k != -1);
         if ((max_k == -1) || (Pcontrib[i].n == 0))
         {
            free(Pcpool);
            free(Pcontrib);
            free(Pcontrib_bounds);
            return NULL;
         }

         if (total_weight != RR(1.0))
            Pcontrib[i].p[max_k].weight += RR(1.0) - total_weight;
      }
   }
   else
   {
      /* Handle case when there are more
      * destination samples than source
      * samples (upsampling).
      */

      half_width = filter_support * filter_scale;

      // Find the source sample(s) that contribute to each destination sample.

      for (i = 0, n = 0; i < dst_x; i++)
      {
         // Convert from discrete to continuous coordinates, scale, then convert back to discrete.
         center = ((Resample_Real)i + NUDGE) / xscale;
         center -= NUDGE;
         center += src_ofs;

         left   = cast_to_int(std::floor(center - half_width));
         right  = cast_to_int(std::ceil(center + half_width));

         Pcontrib_bounds[i].center = center;
         Pcontrib_bounds[i].left		= left;
         Pcontrib_bounds[i].right	= right;

         n += (right - left + 1);
      }

      /* Allocate memory for contributors. */

      int total = n;
      if ((total == 0) || ((Pcpool = (Contrib*)calloc(total, sizeof(Contrib))) == NULL))
      {
         free(Pcontrib);
         free(Pcontrib_bounds);
         return NULL;
      }

      Pcpool_next = Pcpool;

      /* Create the list of source samples which
      * contribute to each destination sample.
      */

      for (i = 0; i < dst_x; i++)
      {
         int max_k = -1;
         Resample_Real max_w = RR(-1e+20);

         center = Pcontrib_bounds[i].center;
         left   = Pcontrib_bounds[i].left;
         right  = Pcontrib_bounds[i].right;

         Pcontrib[i].n = 0;
         Pcontrib[i].p = Pcpool_next;
         Pcpool_next += (right - left + 1);
         resampler_assert((Pcpool_next - Pcpool) <= total);

         total_weight = 0;
         for (j = left; j <= right; j++)
            total_weight += (*Pfilter)((center - (Resample_Real)j) * oo_filter_scale);

         const Resample_Real norm = static_cast<Resample_Real>(RR(1.0) / total_weight);

         total_weight = 0;

#if RESAMPLER_DEBUG
         printf("%i: ", i);
#endif

         for (j = left; j <= right; j++)
         {
            weight = (*Pfilter)((center - (Resample_Real)j) * oo_filter_scale) * norm;
            if (weight == RR(0.0))
               continue;

            n = reflect(j, src_x, boundary_op);

#if RESAMPLER_DEBUG
            printf("%i(%f), ", n, weight);
#endif

            /* Increment the number of source
            * samples which contribute to the
            * current destination sample.
            */

            k = Pcontrib[i].n++;

            Pcontrib[i].p[k].pixel  = (unsigned short)(n);       /* store src sample number */
            Pcontrib[i].p[k].weight = weight; /* store src sample weight */

            total_weight += weight;          /* total weight of all contributors */

            if (weight > max_w)
            {
               max_w = weight;
               max_k = k;
            }
         }

#if RESAMPLER_DEBUG
         printf("\n\n");
#endif

         //resampler_assert(Pcontrib[i].n);
         //resampler_assert(max_k != -1);

         if ((max_k == -1) || (Pcontrib[i].n == 0))
         {
            free(Pcpool);
            free(Pcontrib);
            free(Pcontrib_bounds);
            return NULL;
         }

         if (total_weight != RR(1.0))
            Pcontrib[i].p[max_k].weight += RR(1.0) - total_weight;
      }
   }

#if RESAMPLER_DEBUG
   printf("*******\n");
#endif

   free(Pcontrib_bounds);

   return Pcontrib;
}

void Resampler::resample_x(Sample* Pdst, const Sample* Psrc)
{
   resampler_assert(Pdst);
   resampler_assert(Psrc);

   int i, j;
   Sample total;
   Contrib_List *Pclist = m_Pclist_x;
   Contrib *p;

   for (i = m_resample_dst_x; i > 0; i--, Pclist++)
   {
#if RESAMPLER_DEBUG_OPS
      total_ops += Pclist->n;
#endif

      for (j = Pclist->n, p = Pclist->p, total = 0; j > 0; j--, p++)
         total += Psrc[p->pixel] * p->weight;

      *Pdst++ = total;
   }
}

void Resampler::scale_y_mov(Sample* Ptmp, const Sample* Psrc, Resample_Real weight, int dst_x)
{
   int i;

#if RESAMPLER_DEBUG_OPS
   total_ops += dst_x;
#endif

   // Not += because temp buf wasn't cleared.
   for (i = dst_x; i > 0; i--)
      *Ptmp++ = *Psrc++ * weight;
}

void Resampler::scale_y_add(Sample* Ptmp, const Sample* Psrc, Resample_Real weight, int dst_x)
{
#if RESAMPLER_DEBUG_OPS
   total_ops += dst_x;
#endif

   for (int i = dst_x; i > 0; i--)
      (*Ptmp++) += *Psrc++ * weight;
}

void Resampler::clamp(Sample* Pdst, int n)
{
   while (n > 0)
   {
      *Pdst = clamp_sample(*Pdst);
      ++Pdst;
      n--;
   }
}

void Resampler::resample_y(Sample* Pdst)
{
   int i, j;
   Sample* Psrc;
   Contrib_List* Pclist = &m_Pclist_y[m_cur_dst_y];

   Sample* Ptmp = m_delay_x_resample ? m_Ptmp_buf : Pdst;
   resampler_assert(Ptmp);

   /* Process each contributor. */

   for (i = 0; i < Pclist->n; i++)
   {
      /* locate the contributor's location in the scan
      * buffer -- the contributor must always be found!
      */

      for (j = 0; j < MAX_SCAN_BUF_SIZE; j++)
         if (m_Pscan_buf->scan_buf_y[j] == Pclist->p[i].pixel)
            break;

      resampler_assert(j < MAX_SCAN_BUF_SIZE);

      Psrc = m_Pscan_buf->scan_buf_l[j];

      if (!i)
         scale_y_mov(Ptmp, Psrc, Pclist->p[i].weight, m_intermediate_x);
      else
         scale_y_add(Ptmp, Psrc, Pclist->p[i].weight, m_intermediate_x);

      /* If this source line doesn't contribute to any
      * more destination lines then mark the scanline buffer slot
      * which holds this source line as free.
      * (The max. number of slots used depends on the Y
      * axis sampling factor and the scaled filter width.)
      */

      if (--m_Psrc_y_count[resampler_range_check(Pclist->p[i].pixel, m_resample_src_y)] == 0)
      {
         m_Psrc_y_flag[resampler_range_check(Pclist->p[i].pixel, m_resample_src_y)] = FALSE;
         m_Pscan_buf->scan_buf_y[j] = -1;
      }
   }

   /* Now generate the destination line */

   if (m_delay_x_resample) // Was X resampling delayed until after Y resampling?
   {
      resampler_assert(Pdst != Ptmp);
      resample_x(Pdst, Ptmp);
   }
   else
   {
      resampler_assert(Pdst == Ptmp);
   }

   if (m_lo < m_hi)
      clamp(Pdst, m_resample_dst_x);
}

bool Resampler::put_line(const Sample* Psrc)
{
   int i;

   if (m_cur_src_y >= m_resample_src_y)
      return false;

   /* Does this source line contribute
   * to any destination line? if not,
   * exit now.
   */

   if (!m_Psrc_y_count[resampler_range_check(m_cur_src_y, m_resample_src_y)])
   {
      m_cur_src_y++;
      return true;
   }

   /* Find an empty slot in the scanline buffer. (FIXME: Perf. is terrible here with extreme scaling ratios.) */

   for (i = 0; i < MAX_SCAN_BUF_SIZE; i++)
      if (m_Pscan_buf->scan_buf_y[i] == -1)
         break;

   /* If the buffer is full, exit with an error. */

   if (i == MAX_SCAN_BUF_SIZE)
   {
      m_status = STATUS_SCAN_BUFFER_FULL;
      return false;
   }

   m_Psrc_y_flag[resampler_range_check(m_cur_src_y, m_resample_src_y)] = TRUE;
   m_Pscan_buf->scan_buf_y[i]  = m_cur_src_y;

   /* Does this slot have any memory allocated to it? */

   if (!m_Pscan_buf->scan_buf_l[i])
   {
      if ((m_Pscan_buf->scan_buf_l[i] = (Sample*)malloc(m_intermediate_x * sizeof(Sample))) == NULL)
      {
         m_status = STATUS_OUT_OF_MEMORY;
         return false;
      }
   }

   // Resampling on the X axis first?
   if (m_delay_x_resample)
   {
      resampler_assert(m_intermediate_x == m_resample_src_x);

      // Y-X resampling order
      memcpy(m_Pscan_buf->scan_buf_l[i], Psrc, m_intermediate_x * sizeof(Sample));
   }
   else
   {
      resampler_assert(m_intermediate_x == m_resample_dst_x);

      // X-Y resampling order
      resample_x(m_Pscan_buf->scan_buf_l[i], Psrc);
   }

   m_cur_src_y++;

   return true;
}

const Resampler::Sample* Resampler::get_line()
{
   int i;

   /* If all the destination lines have been
   * generated, then always return NULL.
   */

   if (m_cur_dst_y == m_resample_dst_y)
      return NULL;

   /* Check to see if all the required
   * contributors are present, if not,
   * return NULL.
   */

   for (i = 0; i < m_Pclist_y[m_cur_dst_y].n; i++)
      if (!m_Psrc_y_flag[resampler_range_check(m_Pclist_y[m_cur_dst_y].p[i].pixel, m_resample_src_y)])
         return NULL;

   resample_y(m_Pdst_buf);

   m_cur_dst_y++;

   return m_Pdst_buf;
}

Resampler::~Resampler()
{
   int i;

#if RESAMPLER_DEBUG_OPS
   printf("actual ops: %i\n", total_ops);
#endif

   free(m_Pdst_buf);
   m_Pdst_buf = NULL;

   if (m_Ptmp_buf)
   {
      free(m_Ptmp_buf);
      m_Ptmp_buf = NULL;
   }

   /* Don't deallocate a contibutor list
   * if the user passed us one of their own.
   */

   if ((m_Pclist_x) && (!m_clist_x_forced))
   {
      free(m_Pclist_x->p);
      free(m_Pclist_x);
      m_Pclist_x = NULL;
   }

   if ((m_Pclist_y) && (!m_clist_y_forced))
   {
      free(m_Pclist_y->p);
      free(m_Pclist_y);
      m_Pclist_y = NULL;
   }

   free(m_Psrc_y_count);
   m_Psrc_y_count = NULL;

   free(m_Psrc_y_flag);
   m_Psrc_y_flag = NULL;

   if (m_Pscan_buf)
   {
      for (i = 0; i < MAX_SCAN_BUF_SIZE; i++)
         free(m_Pscan_buf->scan_buf_l[i]);

      free(m_Pscan_buf);
      m_Pscan_buf = NULL;
   }
}

void Resampler::restart()
{
   if (STATUS_OKAY != m_status)
      return;

   m_cur_src_y = m_cur_dst_y = 0;

   int i, j;
   for (i = 0; i < m_resample_src_y; i++)
   {
      m_Psrc_y_count[i] = 0;
      m_Psrc_y_flag[i] = FALSE;
   }

   for (i = 0; i < m_resample_dst_y; i++)
   {
      for (j = 0; j < m_Pclist_y[i].n; j++)
         m_Psrc_y_count[resampler_range_check(m_Pclist_y[i].p[j].pixel, m_resample_src_y)]++;
   }

   for (i = 0; i < MAX_SCAN_BUF_SIZE; i++)
   {
      m_Pscan_buf->scan_buf_y[i] = -1;

      free(m_Pscan_buf->scan_buf_l[i]);
      m_Pscan_buf->scan_buf_l[i] = NULL;
   }
}

Resampler::Resampler(int src_x, int src_y,
                     int dst_x, int dst_y,
                     Boundary_Op boundary_op,
                     Resample_Real sample_low, Resample_Real sample_high,
                     const char* Pfilter_name,
                     Contrib_List* Pclist_x,
                     Contrib_List* Pclist_y,
                     Resample_Real filter_x_scale,
                     Resample_Real filter_y_scale,
                     Resample_Real src_x_ofs,
                     Resample_Real src_y_ofs)
{
   int i, j;
   Resample_Real support, (*func)(Resample_Real);

   resampler_assert(src_x > 0);
   resampler_assert(src_y > 0);
   resampler_assert(dst_x > 0);
   resampler_assert(dst_y > 0);

#if RESAMPLER_DEBUG_OPS
   total_ops = 0;
#endif

   m_lo = sample_low;
   m_hi = sample_high;

   m_delay_x_resample = false;
   m_intermediate_x = 0;
   m_Pdst_buf = NULL;
   m_Ptmp_buf = NULL;
   m_clist_x_forced = false;
   m_Pclist_x = NULL;
   m_clist_y_forced = false;
   m_Pclist_y = NULL;
   m_Psrc_y_count = NULL;
   m_Psrc_y_flag = NULL;
   m_Pscan_buf = NULL;
   m_status = STATUS_OKAY;

   m_resample_src_x = src_x;
   m_resample_src_y = src_y;
   m_resample_dst_x = dst_x;
   m_resample_dst_y = dst_y;

   m_boundary_op = boundary_op;

   if ((m_Pdst_buf = (Sample*)malloc(m_resample_dst_x * sizeof(Sample))) == NULL)
   {
      m_status = STATUS_OUT_OF_MEMORY;
      return;
   }

   // Find the specified filter.

   if (Pfilter_name == NULL)
      Pfilter_name = RESAMPLER_DEFAULT_FILTER;

   for (i = 0; i < NUM_FILTERS; i++)
      if (strcmp(Pfilter_name, g_filters[i].name) == 0)
         break;

   if (i == NUM_FILTERS)
   {
      m_status = STATUS_BAD_FILTER_NAME;
      return;
   }

   func = g_filters[i].func;
   support = g_filters[i].support;

   /* Create contributor lists, unless the user supplied custom lists. */

   if (!Pclist_x)
   {
      m_Pclist_x = make_clist(m_resample_src_x, m_resample_dst_x, m_boundary_op, func, support, filter_x_scale, src_x_ofs);
      if (!m_Pclist_x)
      {
         m_status = STATUS_OUT_OF_MEMORY;
         return;
      }
   }
   else
   {
      m_Pclist_x = Pclist_x;
      m_clist_x_forced = true;
   }

   if (!Pclist_y)
   {
      m_Pclist_y = make_clist(m_resample_src_y, m_resample_dst_y, m_boundary_op, func, support, filter_y_scale, src_y_ofs);
      if (!m_Pclist_y)
      {
         m_status = STATUS_OUT_OF_MEMORY;
         return;
      }
   }
   else
   {
      m_Pclist_y = Pclist_y;
      m_clist_y_forced = true;
   }

   if ((m_Psrc_y_count = (int*)calloc(m_resample_src_y, sizeof(int))) == NULL)
   {
      m_status = STATUS_OUT_OF_MEMORY;
      return;
   }

   if ((m_Psrc_y_flag = (unsigned char*)calloc(m_resample_src_y, sizeof(unsigned char))) == NULL)
   {
      m_status = STATUS_OUT_OF_MEMORY;
      return;
   }

   /* Count how many times each source line
   * contributes to a destination line.
   */

   for (i = 0; i < m_resample_dst_y; i++)
      for (j = 0; j < m_Pclist_y[i].n; j++)
         m_Psrc_y_count[resampler_range_check(m_Pclist_y[i].p[j].pixel, m_resample_src_y)]++;

   if ((m_Pscan_buf = (Scan_Buf*)malloc(sizeof(Scan_Buf))) == NULL)
   {
      m_status = STATUS_OUT_OF_MEMORY;
      return;
   }

   for (i = 0; i < MAX_SCAN_BUF_SIZE; i++)
   {
      m_Pscan_buf->scan_buf_y[i] = -1;
      m_Pscan_buf->scan_buf_l[i] = NULL;
   }

   m_cur_src_y = m_cur_dst_y = 0;
   {
      // Determine which axis to resample first by comparing the number of multiplies required
      // for each possibility.
      int x_ops = count_ops(m_Pclist_x, m_resample_dst_x);
      int y_ops = count_ops(m_Pclist_y, m_resample_dst_y);

      // Hack 10/2000: Weight Y axis ops a little more than X axis ops.
      // (Y axis ops use more cache resources.)
      int xy_ops = x_ops * m_resample_src_y +
         (4 * y_ops * m_resample_dst_x)/3;

      int yx_ops = (4 * y_ops * m_resample_src_x)/3 +
         x_ops * m_resample_dst_y;

#if RESAMPLER_DEBUG_OPS
      printf("src: %i %i\n", m_resample_src_x, m_resample_src_y);
      printf("dst: %i %i\n", m_resample_dst_x, m_resample_dst_y);
      printf("x_ops: %i\n", x_ops);
      printf("y_ops: %i\n", y_ops);
      printf("xy_ops: %i\n", xy_ops);
      printf("yx_ops: %i\n", yx_ops);
#endif

      // Now check which resample order is better. In case of a tie, choose the order
      // which buffers the least amount of data.
      if ((xy_ops > yx_ops) ||
         ((xy_ops == yx_ops) && (m_resample_src_x < m_resample_dst_x))
         )
      {
         m_delay_x_resample = true;
         m_intermediate_x = m_resample_src_x;
      }
      else
      {
         m_delay_x_resample = false;
         m_intermediate_x = m_resample_dst_x;
      }
#if RESAMPLER_DEBUG_OPS
      printf("delaying: %i\n", m_delay_x_resample);
#endif
   }

   if (m_delay_x_resample)
   {
      if ((m_Ptmp_buf = (Sample*)malloc(m_intermediate_x * sizeof(Sample))) == NULL)
      {
         m_status = STATUS_OUT_OF_MEMORY;
         return;
      }
   }
}

void Resampler::get_clists(Contrib_List** ptr_clist_x, Contrib_List** ptr_clist_y)
{
   if (ptr_clist_x)
      *ptr_clist_x = m_Pclist_x;

   if (ptr_clist_y)
      *ptr_clist_y = m_Pclist_y;
}

int Resampler::get_filter_num()
{
   return NUM_FILTERS;
}

char* Resampler::get_filter_name(int filter_num)
{
   if ((filter_num < 0) || (filter_num >= NUM_FILTERS))
      return NULL;
   else
      return g_filters[filter_num].name;
}


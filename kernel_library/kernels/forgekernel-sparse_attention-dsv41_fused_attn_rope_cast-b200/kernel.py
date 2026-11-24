"""ForgeKernel sparse_attention-dsv41_fused_attn_rope_cast-b200 — reference bundle

Reference bundle — kernel.py JIT-builds sibling CUDA sources from disk and pins a code-generation target; the mechanical one-file rule cannot rewrite this loader shape"""

_SOURCES = {
    'dsa.cu': r"""// DeepSeek-V4.1 prefill: Q RoPE + sparse MQA attention (sink) + conjugate O RoPE + per-32 e4m3 cast, sm_100a.
//
// A CTA pair handles one query token at a time (persistent; static tokens cl + k * ncl, then a dynamically claimed
// tail). CTA r owns heads 64r..64r+63 and gathers dims 256r..256r+255 of every selected kv row (cp.async into 128-B
// swizzled tiles of 64 keys), so each kv byte is read once per pair. Attention runs over units of 128 keys (2 tiles).
//   QK: tcgen05 2-CTA, M=128, N=128 per tile, A = Q from TMEM. In this shape TMEM lanes 0..63 of a CTA meet the even
//       CTA's B half and lanes 64..127 the odd CTA's; lanes l / 64+l hold Q[head l] dims 0..255 / 256..511, so they
//       receive the two partial dot products of head l with the unit's keys. The softmax threads add them through smem
//       (two-phase exchange; phase A of the next unit is written right after the P store of the current one).
//   PV: 2-CTA, M=128, N=256 per instruction (2 per K step), A = P (smem), B = the same tile read MN-major.
//       O: lane l = dims 0..255, lane 64+l = dims 256..511 of head l.
// TMEM: O 256 cols | S 128 cols | Q 128 cols.
// Warps: 0..3, 6, 7 gather (pairs take the tiles G % 3), 4 QK issuer (even CTA) / tile-full relay (odd CTA),
//        5 PV issuer (even CTA) / tail-token claimer (odd CTA), 8..11 softmax (one warp per TMEM lane quadrant),
//        12..19 token warps (Q -> TMEM, cos/sin, epilogue).
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include "dsa_ptx.cuh"

namespace dsa {

constexpr int HQ = 128, TOPK = 640;
constexpr int UK = 128;                // keys per attention unit (two tiles)
constexpr int TK = 64;                 // keys per gathered tile
constexpr int NS = 6;                  // tile ring slots
constexpr int SLOT = TK * 512;         // 64 keys x 256 dims x 2 B
constexpr int NGW = 4;                 // gather warps: key half w & 1 of the units U = w >> 1 mod 2
constexpr int NTHR = 640;              // 20 warps

constexpr int OFF_RING = 0;
constexpr int OFF_P = OFF_RING + NS * SLOT;  // P [64 heads x 128 keys] bf16 as 2 SW128 blocks of 64 keys
constexpr int OFF_X = OFF_P + 2 * 8192;      // partial-score exchange (two phases) [8][128] x 16 B
constexpr int OFF_MASK = OFF_X + 16384;      // [NS][2] validity words of each ring tile
constexpr int OFF_CS = OFF_MASK + 64;        // [3][64] cos/sin
constexpr int OFF_ST = OFF_CS + 768;         // [3][128] row stats (m, l, raw max) of one token
constexpr int OFF_BAR = OFF_ST + 1536;
constexpr int NBAR = 48;
constexpr int OFF_TPTR = OFF_BAR + NBAR * 8;  // [0] TMEM address, [1] (odd CTA) token index of the token warps
constexpr int OFF_TSEQ = OFF_TPTR + 16;        // [NTSEQ] this pair's tokens in order (-1: tail token not claimed yet)
constexpr int NTSEQ = 64;
constexpr int SMEM = OFF_TSEQ + 4 * NTSEQ;
// Token schedule: pair cl takes the tokens cl + k * ncl below SSTAT = (s_q / ncl - TAIL) * ncl; the rest (the tail) is
// claimed dynamically from a global counter, so the pairs that run ahead (SM speeds differ by a few %) take more.
constexpr int TAIL = 1;
static_assert(SMEM <= 232448, "smem");

enum {
  FULL = 0,                // local tile full (gather threads' cp.async arrivals + validity words)
  PEERFULL = FULL + NS,    // (even CTA) the odd CTA's tile is full (relayed)
  EMPTY = PEERFULL + NS,   // tile consumed (MMA commit)
  SFULL = EMPTY + NS,      // S written (MMA commit)
  SFREE,                   // (even CTA) S read by both CTAs' softmax
  PFULL,                   // (even CTA) P written by both CTAs' softmax
  PVDONE,                  // PV done (MMA commit)
  QKLAST,                  // last QK of a token done: Q region free
  QREADY,                  // (even CTA) next Q stored by both CTAs
  OFULL,                   // last PV of a token done
  OFREE,                   // (even CTA) O drained by both CTAs
  STATS,
  STEMPTY = STATS + 2,
  NB = STEMPTY + 2
};
static_assert(NB <= NBAR, "barriers");

constexpr uint32_t COL_O = 0, COL_S = 256, COL_Q = 384;
constexpr uint32_t IDESC_QK = px::idesc_bf16(128, 128, 0, 0);
constexpr uint32_t IDESC_PV = px::idesc_bf16(128, 256, 0, 1);
constexpr float LOG2E = 1.4426950408889634f, LN2 = 0.6931471805599453f;
constexpr float RESCALE = 8.f;  // lazy rescale threshold (log2 units)

struct Params {
  const uint4* q;  // [s_q][8192] 16-B pieces (Q_PERM order)
  const char* kv;
  long long kv_row;  // bytes
  const int* idx;
  long long idx_stride;
  const int* pos;
  const float* cs;
  const float* sink;
  const int* tlen;
  uint8_t* out;
  int* sf;
  float* ml;
  float* lse;
  int s_q, s_kv;
  float scale;
  int dbg;
  long long* trace;  // debug timeline of CTA 0 (null: off)
};
#ifdef DSA_TRACE
#define TR(ev, n) do { if (blockIdx.x == 0 && p.trace && (n) < 1024) p.trace[(n) * 8 + (ev)] = clock64(); } while (0)
#define TR2(ev, n) do { if (blockIdx.x == 0 && p.trace && (n) < 1024) p.trace[8192 + (n) * 8 + (ev)] = clock64(); } while (0)
#define TR3(ev, n) do { if (blockIdx.x == 0 && p.trace && (n) < 512) p.trace[16384 + (n) * 8 + (ev)] = clock64(); } while (0)
#define TR4(ev, n) do { if (blockIdx.x == 0 && p.trace && (n) < 1024) p.trace[20480 + (n) * 8 + (ev)] = clock64(); } while (0)
#else
#define TR4(ev, n) do {} while (0)
#define TR3(ev, n) do {} while (0)
#define TR(ev, n) do {} while (0)
#define TR2(ev, n) do {} while (0)
#endif

// arrive on barrier k of the even CTA (local on rank 0; relaxed remote arrive from rank 1)
#define ARRIVE0(k) do { if (rank == 0) px::bar_arrive(bar(k)); else px::bar_arrive_cl_relaxed(bar0(k)); } while (0)

__device__ __forceinline__ int lane_id() { return threadIdx.x & 31; }
__device__ __forceinline__ int topk_len(const Params& p, int t) {
  return p.tlen ? min(max(p.tlen[t], 0), TOPK) : TOPK;
}
__device__ __forceinline__ int nunits(const Params& p, int t) { return max(1, (topk_len(p, t) + UK - 1) / UK); }
__device__ __forceinline__ int ntiles(const Params& p, int t) { return 2 * nunits(p, t); }

__device__ int g_tail[2];  // [0] tail tokens claimed, [1] pairs done claiming (the last one resets both)
// k-th token of this pair (>= s_q: none)
__device__ __forceinline__ int tseq(uint32_t sb, int k) {
  int v;
  do {
    asm volatile("ld.volatile.shared.b32 %0, [%1];" : "=r"(v) : "r"(sb + OFF_TSEQ + 4 * min(k, NTSEQ - 1)));
  } while (v < 0);
  return v;
}

// ---------------------------------------------------------------------------------------------------- softmax
// One group of 4 warps (one per TMEM lane quadrant) handles every 128-key unit (two tiles). Thread (q, lane): lane
// i = 32q + lane, head h = i % 64, dims half kh = i / 64 owns the keys of tile kh: it loads its lane's partial sums of
// both tiles, hands the partner (lane i ^ 64) those of the partner's tile and adds the partner's to its own. The
// running max m (log2 units, lazily raised by more than RESCALE) is kept per thread (identical within a pair).
__device__ __forceinline__ void softmax_role(const Params& p, uint8_t* smem) {
  const int q = (threadIdx.x >> 5) & 3, lane = threadIdx.x & 31;
  const uint32_t sb = px::saddr(smem), tmem = *(volatile uint32_t*)(smem + OFF_TPTR);
  const uint32_t rank = px::cta_rank();
  const int cl = px::pair_index(), ncl = px::pair_count();
  const int i = 32 * q + lane, h = i & 63, kh = i >> 6;
  const uint32_t tl = tmem + ((uint32_t)(32 * q) << 16);
  auto bar = [&](int n) { return sb + OFF_BAR + 8 * n; };
  auto bar0 = [&](int n) { return px::peer_addr(bar(n), 0); };
  // exchange slot of thread j: 8 chunks of 16 B at stride 2 KB (consecutive threads -> consecutive 16 B). Phase A:
  // keys 0..31 go to the partner's slot; phase B: keys 32..63 to this thread's own (already read) slot; the unit
  // maxima then go to the partner's slot (read by now)
  uint4* xs_p = (uint4*)(smem + OFF_X) + (i ^ 64);
  uint4* xs_o = (uint4*)(smem + OFF_X) + i;
  const volatile uint32_t* mask = (const volatile uint32_t*)(smem + OFF_MASK);
  float* st = (float*)(smem + OFF_ST);
  const float cl2 = p.scale * LOG2E;
  const int pbar = 1 + (q & 1);
  // P [64 heads][128 keys]: block kh (keys 64kh..) row h, 8 chunks of 16 B swizzled with h % 8
  uint8_t* prow = smem + OFF_P + kh * 8192 + h * 128;
  const uint32_t a_sfree = rank ? bar0(SFREE) : bar(SFREE), a_pfull = rank ? bar0(PFULL) : bar(PFULL);
  auto arrive0 = [&](uint32_t a) {
    if (rank) px::bar_arrive_cl_relaxed(a);
    else px::bar_arrive(a);
  };
#ifdef DSA_TRACE
  long long pacc[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, c_prev = 0;
#define PACC(j) do { const long long c_ = clock64(); pacc[j] += c_ - c_prev; c_prev = c_; } while (0)
#else
#define PACC(j) do {} while (0)
#endif
  // Software pipelined over units: during the exponentials of unit U the partial sums of unit U + 1 (the partner's
  // tile) are loaded, and phase A of its exchange plus the load of this thread's own tile are issued before the P
  // stores of U, so the next unit starts with its S in registers.
  uint32_t U = 0;
  int tc = 0;
  float v[64];
  uint32_t sh[32];  // partner-tile partial sums 32..63 of the next unit (phase B of its exchange)
  if (tseq(sb, 0) < p.s_q) {
    uint32_t so[32];
    px::bar_wait(bar(SFULL), 0);
    px::tc_after();
    px::tm_ld32(tl + COL_S + 64 * (1 - kh), so);
    px::tm_ld32(tl + COL_S + 64 * (1 - kh) + 32, sh);
    px::tm_ld_wait();
#pragma unroll
    for (int c = 0; c < 8; ++c) xs_p[c * 128] = make_uint4(so[4 * c], so[4 * c + 1], so[4 * c + 2], so[4 * c + 3]);
    px::tm_ld32(tl + COL_S + 64 * kh, (uint32_t*)v);
    px::tm_ld32(tl + COL_S + 64 * kh + 32, (uint32_t*)v + 32);
  }
  float m_carry = -INFINITY;
  for (int t = tseq(sb, 0), tn; t < p.s_q; t = tn, ++tc) {
    tn = tseq(sb, tc + 1);  // (claimed by now if a tail token)
    const int nu = nunits(p, t);
    // the running max starts from the previous token's (same row, identical in the pair): the first unit then needs
    // no hand-off of maxima unless the new token's max is far above or below it
    float m_g = m_carry, l_run = 0.f, rmax = -INFINITY;
    for (int u = 0; u < nu; ++u, ++U) {
      PACC(0);
      const uint32_t G = 2 * U + kh, ts = G % NS, tph = (G / NS) & 1;  // this thread's tile, ring slot, FULL parity
      const bool nxt = u + 1 < nu || tn < p.s_q;
      uint32_t mw0, mw1;
      px::tm_ld_wait();
      px::tc_before();
      __syncwarp();
      if (px::elect()) arrive0(a_sfree);
      PACC(1);
      // the tile's validity words: written and released before its FULL arrive, which the QK issuer observed
      // before issuing QK(U) (whose completion we have observed)
      mw0 = mask[2 * ts];
      mw1 = mask[2 * ts + 1];
      px::nbar(pbar, 64);
      PACC(2);
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        const uint4 w = xs_o[c * 128];
        px::up2(px::add2(px::pk2(v[4 * c], v[4 * c + 1]), px::pk2(__uint_as_float(w.x), __uint_as_float(w.y))),
                v[4 * c], v[4 * c + 1]);
        px::up2(px::add2(px::pk2(v[4 * c + 2], v[4 * c + 3]), px::pk2(__uint_as_float(w.z), __uint_as_float(w.w))),
                v[4 * c + 2], v[4 * c + 3]);
      }
#pragma unroll
      for (int c = 0; c < 8; ++c) xs_o[c * 128] = make_uint4(sh[4 * c], sh[4 * c + 1], sh[4 * c + 2], sh[4 * c + 3]);
      px::nbar(pbar, 64);
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        const uint4 w = xs_p[c * 128];
        px::up2(px::add2(px::pk2(v[32 + 4 * c], v[33 + 4 * c]), px::pk2(__uint_as_float(w.x), __uint_as_float(w.y))),
                v[32 + 4 * c], v[33 + 4 * c]);
        px::up2(px::add2(px::pk2(v[34 + 4 * c], v[35 + 4 * c]), px::pk2(__uint_as_float(w.z), __uint_as_float(w.w))),
                v[34 + 4 * c], v[35 + 4 * c]);
      }
      if ((mw0 & mw1) != 0xffffffffu) {
#pragma unroll
        for (int e = 0; e < 32; ++e) {
          v[e] = ((mw0 >> e) & 1) ? v[e] : -INFINITY;
          v[32 + e] = ((mw1 >> e) & 1) ? v[32 + e] : -INFINITY;
        }
      }
      float mx;
      {
        float m22[22];
#pragma unroll
        for (int e = 0; e < 21; ++e) m22[e] = fmaxf(fmaxf(v[3 * e], v[3 * e + 1]), v[3 * e + 2]);
        m22[21] = v[63];
        float m8[8];
#pragma unroll
        for (int e = 0; e < 7; ++e) m8[e] = fmaxf(fmaxf(m22[3 * e], m22[3 * e + 1]), m22[3 * e + 2]);
        m8[7] = m22[21];
        const float a = fmaxf(fmaxf(m8[0], m8[1]), m8[2]), b = fmaxf(fmaxf(m8[3], m8[4]), m8[5]);
        mx = fmaxf(fmaxf(a, b), fmaxf(m8[6], m8[7]));
      }
      // the running max (identical in the pair) is raised, lazily by more than RESCALE, to the pair's unit max only if
      // one thread of the pair sees a key above the threshold: the common case costs one reducing barrier and no
      // exchange of maxima (rmax then holds this thread's own keys; the token warps combine the pair)
      const float mp = m_g;
      float m_new = mp;
      const float xs0 = mx * cl2;
      if (px::nbar_or(pbar, 64, xs0 > mp + RESCALE || (u == 0 && !(xs0 >= mp - 20.f)))) {
        *(float*)xs_p = mx;  // the partner's slot: its phase-B data has been read
        px::nbar(pbar, 64);
        mx = fmaxf(mx, *(const volatile float*)xs_o);
        const float xs = mx * cl2;
        m_new = u == 0 || xs > mp + RESCALE ? xs : mp;  // (first unit: O is overwritten, any m is fine)
        px::nbar(pbar, 64);  // both maxima read before the next unit's phase A overwrites the slots
      }
      rmax = fmaxf(rmax, mx);
      PACC(3);
      bool rescaled = false;
      const bool fix = u > 0 && m_new > mp && mp != -INFINITY;
      if (__any_sync(0xffffffffu, fix)) {
        // O holds this token's partial sums at scale mp: wait for PV(U-1), then rescale this lane's 256 columns
        px::bar_wait(bar(PVDONE), (U - 1) & 1);
        px::tc_after();
        const float alpha = fix ? px::exp2a(mp - m_new) : 1.f;
#pragma unroll 1
        for (int c = 0; c < 256; c += 32) {
          uint32_t r[32];
          px::tm_ld32(tl + COL_O + c, r);
          px::tm_ld_wait();
#pragma unroll
          for (int e = 0; e < 32; ++e) r[e] = __float_as_uint(__uint_as_float(r[e]) * alpha);
          px::tm_st32(tl + COL_O + c, r);
        }
        px::tm_st_wait();
        rescaled = true;
      }
      if (fix) l_run *= px::exp2a(mp - m_new);
      m_g = m_new;
      const float mu = m_new == -INFINITY ? 0.f : m_new;
      const uint64_t c2 = px::pk2(cl2, cl2), nm2 = px::pk2(-mu, -mu);
      uint32_t pk[32];
      uint64_t ls = 0, ls2 = 0;
#pragma unroll
      for (int e = 0; e < 32; ++e) {
        float a, bb;
        px::up2(px::fma2(px::pk2(v[2 * e], v[2 * e + 1]), c2, nm2), a, bb);
        a = px::exp2a(a);
        bb = px::exp2a(bb);
        if (e & 1) ls2 = px::add2(ls2, px::pk2(a, bb)); else ls = px::add2(ls, px::pk2(a, bb));
        pk[e] = px::bf16x2(a, bb);
      }
      {
        float a0, a1, b0, b1;
        px::up2(ls, a0, a1);
        px::up2(ls2, b0, b1);
        l_run += (a0 + a1) + (b0 + b1);
      }
      PACC(4);
      PACC(5);
      if (U >= 1 && !rescaled) px::bar_wait(bar(PVDONE), (U - 1) & 1);  // P buffer free (PV(U-1) done)
      PACC(6);
#pragma unroll
      for (int j = 0; j < 8; ++j)
        *(uint4*)(prow + ((j ^ (h & 7)) << 4)) = make_uint4(pk[4 * j], pk[4 * j + 1], pk[4 * j + 2], pk[4 * j + 3]);
      px::fence_async_smem();
      if (rescaled) px::tc_before();
      __syncwarp();
      if (px::elect()) arrive0(a_pfull);
      PACC(7);
      if (nxt) {  // the next unit's S: partner-tile sums -> phase A of its exchange (+ sh), own tile -> v
        uint32_t so[32];
        px::bar_wait(bar(SFULL), (U + 1) & 1);
        px::tc_after();
        px::tm_ld32(tl + COL_S + 64 * (1 - kh), so);
        px::tm_ld32(tl + COL_S + 64 * (1 - kh) + 32, sh);
        px::tm_ld_wait();
#pragma unroll
        for (int c = 0; c < 8; ++c) xs_p[c * 128] = make_uint4(so[4 * c], so[4 * c + 1], so[4 * c + 2], so[4 * c + 3]);
        px::tm_ld32(tl + COL_S + 64 * kh, (uint32_t*)v);
        px::tm_ld32(tl + COL_S + 64 * kh + 32, (uint32_t*)v + 32);
      }
      PACC(9);
#ifdef DSA_TRACE
      pacc[8] += 1;
#endif
    }
    // row stats -> the token warps (named barriers 11 / 12 with them: the arrive does not stall this warp)
    if (tc >= 1) px::nbar(12, 384);  // the previous token's stats have been read
    float* sp = st;
    sp[i] = m_g;
    m_carry = m_g;
    sp[128 + i] = l_run;
    sp[256 + i] = rmax;
    px::nbar_arrive(11, 384);
  }
#ifdef DSA_TRACE
  if (p.trace && lane == 0 && blockIdx.x == 0)
    for (int j = 0; j < 10; ++j) p.trace[29696 + 2048 + q * 16 + j] = pacc[j];
#endif
#undef PACC
}

// ---------------------------------------------------------------------------------------------------- token warps
__device__ __forceinline__ void token_role(const Params& p, uint8_t* smem, uint32_t sb, uint32_t tmem, uint32_t rank_,
                                           int cl_, int ncl_, int q, int c, int lane) {
  const uint32_t rank = px::cta_rank();
  const int cl = px::pair_index(), ncl = px::pair_count();
  const int i = 32 * q + lane, h = i & 63, kh = i >> 6, H = 64 * (int)rank + h;
  const uint32_t tl = tmem + ((uint32_t)(32 * q) << 16);
  auto bar = [&](int k) { return sb + OFF_BAR + 8 * k; };
  auto bar0 = [&](int k) { return px::peer_addr(bar(k), 0); };
  const float* cs_s = (const float*)(smem + OFF_CS);
  const float* st = (const float*)(smem + OFF_ST);
  const float sink_l2 = p.sink ? p.sink[H] * LOG2E : -INFINITY;
  const uint64_t pol = px::pol_evict_first();
  const bool rope_q = kh == 1 && c == 1;
  int* const sf_base = p.sf + ((size_t)(H >> 3) * 32 + 2 * ((256 * kh + 128 * c) >> 5) + ((H & 7) >> 2)) * p.s_q;
  const bool poller = q == 0 && c == 0;  // the token warp that polls mbarriers (the others wait in barrier 10)
  // Q of token t: prefetched into L2 a token ahead (one bulk prefetch per CTA), loaded right before the TMEM store
  auto qprefetch = [&](int t) {
    if (poller && lane == 0) px::l2_prefetch(p.q + (size_t)t * 8192 + (size_t)rank * 4096, 65536);
  };
  // Q of token t -> registers (RoPE applied), stored to TMEM once the previous token's QKs are done
  // The loads go in groups of 3, each group's address depending on the previous group's data (through a mask that
  // is 0 at run time, opaque to the compiler): few Q misses in flight next to the gather. The first group of the
  // next token is issued before the epilogue (qf), the rest at the top of its iteration.
  uint4 qf[3];
  auto qsrc = [&](int t) { return p.q + (size_t)t * 8192 + ((((int)rank * 2 + c) * 4 + q) * 16) * 32 + lane; };
  auto qfirst = [&](int t) {
    const uint4* src = qsrc(t);
#pragma unroll
    for (int k = 0; k < 3; ++k) qf[k] = px::ld_stream(src + k * 32, pol);
  };
  auto qload = [&](int t, int slot3, uint32_t* w) {
    const uint4* src = qsrc(t);
    uint4 qr[16];
#pragma unroll
    for (int k = 0; k < 3; ++k) qr[k] = qf[k];
    uint32_t dep = 0;
#pragma unroll
    for (int k = 3; k < 16; ++k) {
      if (k % 3 == 0) dep = (qr[k - 1].x ^ qr[k - 3].y) & ((uint32_t)p.dbg & 0x80000000u);
      qr[k] = px::ld_stream(src + k * 32 + dep, pol);
    }
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      w[4 * k] = qr[k].x; w[4 * k + 1] = qr[k].y; w[4 * k + 2] = qr[k].z; w[4 * k + 3] = qr[k].w;
    }
    if (rope_q) {  // dims 448..511: pair e of the rotated block
      const float* cs = cs_s + slot3 * 64;
#pragma unroll
      for (int e = 0; e < 32; ++e) {
        const float x0 = __uint_as_float(w[32 + e] << 16), x1 = __uint_as_float(w[32 + e] & 0xffff0000u);
        const float co = cs[e], si = cs[32 + e];
        w[32 + e] = px::bf16x2(__fmaf_rn(x0, co, -__fmul_rn(x1, si)), __fmaf_rn(x0, si, __fmul_rn(x1, co)));
      }
    }
  };
  auto qwrite = [&](const uint32_t* w) {
    px::tm_st32(tl + COL_Q + 64 * c, w);
    px::tm_st32(tl + COL_Q + 64 * c + 32, w + 32);
    px::tm_st_wait();
    px::tc_before();
    __syncwarp();
    if (lane == 0) ARRIVE0(QREADY);
  };
  // max |x| over 32 values as a tree (a serial chain is 16 dependent FMNMX3 deep)
  auto amax32 = [](const uint32_t* x) {
    float m11[11];
#pragma unroll
    for (int e = 0; e < 10; ++e)
      m11[e] = fmaxf(fmaxf(fabsf(__uint_as_float(x[3 * e])), fabsf(__uint_as_float(x[3 * e + 1]))),
                     fabsf(__uint_as_float(x[3 * e + 2])));
    m11[10] = fmaxf(fabsf(__uint_as_float(x[30])), fabsf(__uint_as_float(x[31])));
    const float a = fmaxf(fmaxf(m11[0], m11[1]), m11[2]), b = fmaxf(fmaxf(m11[3], m11[4]), m11[5]);
    const float c = fmaxf(fmaxf(m11[6], m11[7]), m11[8]), d = fmaxf(m11[9], m11[10]);
    return fmaxf(fmaxf(a, b), fmaxf(c, d));
  };
  auto epilogue = [&](int t, int tc, bool more) {
    px::nbar(11, 384);  // the softmax's row stats of token tc
    if (i == 0) TR3(5, tc);
    const float* sp = st;
    const float m = sp[i], l = sp[128 + i] + sp[128 + (i ^ 64)];
    const float rmax = fmaxf(sp[256 + i], sp[256 + (i ^ 64)]);
    if (more) px::nbar_arrive(12, 384);  // stats read: the softmax may write the next token's
    const float inv = l > 0.f ? 1.f / (l + px::exp2a(sink_l2 - m)) : 0.f;
    if (kh == 0 && c == 0) {
      p.ml[(size_t)t * HQ + H] = rmax == -INFINITY ? -INFINITY : rmax * p.scale;
      p.lse[(size_t)t * HQ + H] = l > 0.f ? (m + px::log2a(l)) * LN2 : INFINITY;
    }
    const float* cs = cs_s + (tc % 3) * 64;
    uint8_t* orow = p.out + ((size_t)t * 16 + (H >> 3)) * 4096 + 32 * (H & 7);
    if (poller) px::bar_wait(bar(OFULL), tc & 1);
    px::nbar(10, 256);
    px::tc_after();
    if (i == 0) TR3(2, tc);
    // two passes of 64 columns (register budget); O is released after the second load
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      uint32_t o[64];
      px::tm_ld32(tl + COL_O + 128 * c + 64 * half, o);
      px::tm_ld32(tl + COL_O + 128 * c + 64 * half + 32, o + 32);
      px::tm_ld_wait();
      if (half == 1) {
        px::tc_before();
        __syncwarp();
        if (lane == 0) ARRIVE0(OFREE);
        if (i == 0) TR3(3, tc);
      }
      if ((p.dbg & 64) && half == 0) continue;  // debug: skip the first pass (results invalid)
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
      const int j = 2 * half + jj;
      const int d0 = 256 * kh + 128 * c + 32 * j;
      float f;
      int ex;
      if (d0 >= 448) {  // RoPE chunk: rotate (linear: the 1/l scale goes into f as for the other chunks)
#pragma unroll
        for (int e = 0; e < 32; e += 2) {
          const int pp = (d0 + e - 448) >> 1;
          const float co = cs[pp], si = cs[32 + pp];
          const float x0 = __uint_as_float(o[32 * jj + e]), x1 = __uint_as_float(o[32 * jj + e + 1]);
          float r0, r1;
          px::up2(px::fma2(px::pk2(co, -si), px::pk2(x0, x0), px::mul2(px::pk2(si, co), px::pk2(x1, x1))), r0, r1);
          o[32 * jj + e] = __float_as_uint(r0);
          o[32 * jj + e + 1] = __float_as_uint(r1);
        }
        float a = amax32(o + 32 * jj);
        a *= inv;
        const uint32_t bits = __float_as_uint(a);
        ex = max((int)(bits >> 23) - 135 + ((bits & 0x7fffffu) > 0x600000u ? 1 : 0), -127);
        f = inv * __uint_as_float((uint32_t)(127 - ex) << 23);
      } else {
        float a = amax32(o + 32 * jj);
        a *= inv;
        const uint32_t bits = __float_as_uint(a);
        ex = max((int)(bits >> 23) - 135 + ((bits & 0x7fffffu) > 0x600000u ? 1 : 0), -127);
        f = inv * __uint_as_float((uint32_t)(127 - ex) << 23);
      }
      uint32_t w[8];
      const uint64_t f2 = px::pk2(f, f);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        float a0, a1, a2, a3;
        px::up2(px::mul2(px::pk2(__uint_as_float(o[32 * jj + 4 * e]), __uint_as_float(o[32 * jj + 4 * e + 1])), f2), a0, a1);
        px::up2(px::mul2(px::pk2(__uint_as_float(o[32 * jj + 4 * e + 2]), __uint_as_float(o[32 * jj + 4 * e + 3])), f2), a2, a3);
        w[e] = px::e4m3x4(a0, a1, a2, a3);
      }
      uint4* dst = (uint4*)(orow + 256 * (d0 >> 5));
      px::st_stream32(dst, w, pol);
      uint32_t sw = (uint32_t)(ex + 127) << (8 * (H & 3));
      sw |= __shfl_xor_sync(0xffffffffu, sw, 1);
      sw |= __shfl_xor_sync(0xffffffffu, sw, 2);
      if ((H & 3) == 0) sf_base[(size_t)(2 * j) * p.s_q + t] = (int)sw;
    }
    }
  };

  // cos/sin of token tc -> cs_s[tc % 3], written by token warp 8 together with the token's Q prefetch
  float* cs_w = (float*)(smem + OFF_CS);
  auto csload = [&](int t, int tc) {
    if (q == 0 && c == 0) {
      const float* src = p.cs + (size_t)p.pos[t] * 64;
      cs_w[(tc % 3) * 64 + lane] = src[lane];
      cs_w[(tc % 3) * 64 + 32 + lane] = src[32 + lane];
    }
  };
  int tc = 0;
  const int t0 = tseq(sb, 0);
  if (t0 < p.s_q) {
    qprefetch(t0);
    csload(t0, 0);
    qfirst(t0);
  }
  for (int t = t0; t < p.s_q; t = tseq(sb, ++tc)) {
    px::nbar(7, 256);  // cos/sin of this token visible to all token warps
    {
      uint32_t w[64];
      qload(t, tc % 3, w);
      if (tc > 0) {
        if (poller) px::bar_wait(bar(QKLAST), (tc - 1) & 1);
        px::nbar(10, 256);
        px::tc_after();
      }
      if (i == 0) TR3(0, tc);
      qwrite(w);
    }
    if (i == 0) TR3(1, tc);
    const int tn = tseq(sb, tc + 1);
    if (tn < p.s_q) {
      qprefetch(tn);
      qfirst(tn);
    }
    if (tc > 0) epilogue(tseq(sb, tc - 1), tc - 1, true);
    if (i == 0) TR3(4, tc);
    if (rank == 1 && poller && lane == 0) *(volatile int*)(smem + OFF_TPTR + 4) = tc + 1;  // for the tail claimer
    if (tn < p.s_q) csload(tn, tc + 1);
  }
  if (tc > 0) epilogue(tseq(sb, tc - 1), tc - 1, false);
}

// ---------------------------------------------------------------------------------------------------- MMA issuers
// Two issuer warps on the even CTA (the tensor queue is shallow: while one stalls issuing, the other prepares).
// Both run on all 32 lanes (cursors / descriptors warp-uniform); one elected lane issues each batch.
// QK(U) needs tile U, the Q of its token and S buffer U % 2 released by the softmax of U - 2.
__device__ __forceinline__ void qk_role(const Params& p, uint32_t sb, uint32_t tmem, int cl_, int ncl_) {
  const int cl = px::pair_index(), ncl = px::pair_count();
  auto bar = [&](int k) { return sb + OFF_BAR + 8 * k; };
  const uint64_t dq0 = px::sdesc_sw128(sb + OFF_RING, 16, 1024);  // K-major tile
  uint32_t U = 0;
  int tc = 0;
#ifdef DSA_TRACE
  long long qacc[5] = {0, 0, 0, 0, 0};
#endif
  for (int t = tseq(sb, 0); t < p.s_q; t = tseq(sb, ++tc)) {
    const int nu = nunits(p, t);
#ifdef DSA_TRACE
    long long cq = clock64();
#endif
    px::bar_wait(bar(QREADY), tc & 1);
#ifdef DSA_TRACE
    qacc[3] += clock64() - cq;
#endif
    for (int u = 0; u < nu; ++u, ++U) {
#ifdef DSA_TRACE
      long long c0 = clock64();
#endif
      if (U >= 1) px::bar_wait(bar(SFREE), (U - 1) & 1);
#ifdef DSA_TRACE
      long long c1 = clock64(); qacc[0] += c1 - c0;
#endif
      const uint32_t G0 = 2 * U, s0 = G0 % NS, s1 = (G0 + 1) % NS;
#pragma unroll
      for (int j = 0; j < 2; ++j) {  // tile by tile: the first tile's MMAs need not wait for the second tile
        const uint32_t sj = j ? s1 : s0, ph = ((G0 + j) / NS) & 1;
        px::bar_wait(bar(FULL + sj), ph);
        px::bar_wait(bar(PEERFULL + sj), ph);
        px::fence_async_smem();  // cp.async (generic proxy) data -> tensor core
        px::tc_after();
        if (px::elect()) {
          const uint32_t t0 = (sj * SLOT) >> 4;
#pragma unroll
          for (int kk = 0; kk < 16; ++kk)
            px::mma2_ts(tmem + COL_S + 64 * j, tmem + COL_Q + 8 * kk,
                        dq0 + t0 + (((kk >> 2) * 8192 + (kk & 3) * 32) >> 4), IDESC_QK, kk > 0);
        }
        __syncwarp();
      }
      if (px::elect()) {
        px::mma2_commit(bar(SFULL));
        if (u == nu - 1) px::mma2_commit(bar(QKLAST));
      }
      __syncwarp();
    }
  }
#ifdef DSA_TRACE
  if (p.trace && lane_id() == 0 && blockIdx.x < 2)
    for (int j = 0; j < 5; ++j) p.trace[29696 + 2048 + 256 + blockIdx.x * 8 + j] = qacc[j];
#endif
}

// PV(U) needs P(U) (both CTAs' softmax) and, for a token's first unit, the previous token's O drained.
__device__ __forceinline__ void pv_role(const Params& p, uint32_t sb, uint32_t tmem, int cl_, int ncl_) {
  const int cl = px::pair_index(), ncl = px::pair_count();
  auto bar = [&](int k) { return sb + OFF_BAR + 8 * k; };
  const uint64_t dv0 = px::sdesc_sw128(sb + OFF_RING, 8192, 1024);  // MN-major tile
  const uint64_t dp0 = px::sdesc_sw128(sb + OFF_P, 16, 1024);       // P, K-major
#ifdef DSA_TRACE
  long long vacc[3] = {0, 0, 0};
#endif
  uint32_t U = 0;
  int tc = 0;
  for (int t = tseq(sb, 0); t < p.s_q; t = tseq(sb, ++tc)) {
    const int nu = nunits(p, t);
    for (int u = 0; u < nu; ++u, ++U) {
#ifdef DSA_TRACE
      long long c0 = clock64();
#endif
      px::bar_wait(bar(PFULL), U & 1);
#ifdef DSA_TRACE
      long long c1 = clock64(); vacc[0] += c1 - c0;
#endif
      if (u == 0 && tc > 0) px::bar_wait(bar(OFREE), (tc - 1) & 1);
#ifdef DSA_TRACE
      vacc[1] += clock64() - c1; vacc[2] += 1;
#endif
      px::tc_after();
      const uint32_t s0 = (2 * U) % NS, s1 = (2 * U + 1) % NS;
      if (px::elect()) {
        if (!(p.dbg & 2)) {
          // each P K-step feeds both column halves: the second MMA takes A from the collector
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            const uint32_t t0 = ((j ? s1 : s0) * SLOT) >> 4;
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
              px::mma2_ss_afill(tmem + COL_O, dp0 + ((j * 8192 + kk * 32) >> 4), dv0 + t0 + ((kk * 2048) >> 4),
                                IDESC_PV, !(u == 0 && j == 0 && kk == 0));
              px::mma2_ss_alast(tmem + COL_O + 128, dp0 + ((j * 8192 + kk * 32) >> 4),
                                dv0 + t0 + ((16384 + kk * 2048) >> 4), IDESC_PV, !(u == 0 && j == 0 && kk == 0));
            }
            if (j == 0) px::mma2_commit(bar(EMPTY + s0));  // the first tile is free once its MMAs are done
          }
        }
        if (p.dbg & 2) px::mma2_commit(bar(EMPTY + s0));
        px::mma2_commit(bar(PVDONE));
        px::mma2_commit(bar(EMPTY + s1));
        if (u == nu - 1) px::mma2_commit(bar(OFULL));
      }
      __syncwarp();
    }
  }
#ifdef DSA_TRACE
  if (p.trace && lane_id() == 0 && blockIdx.x == 0)
    for (int j = 0; j < 3; ++j) p.trace[29696 + 2048 + 300 + j] = vacc[j];
#endif
}

// ---------------------------------------------------------------------------------------------------- kernel
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(NTHR, 1) dsa_prefill(const Params p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  const uint32_t sb = px::saddr(smem);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const uint32_t rank = px::cta_rank();
  const int cl = blockIdx.x >> 1, ncl = gridDim.x >> 1;
  auto bar = [&](int k) { return sb + OFF_BAR + 8 * k; };
  auto bar0 = [&](int k) { return px::peer_addr(bar(k), 0); };

#ifdef DSA_TRACE
  if (p.trace && threadIdx.x == 0) {
    uint64_t g;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(g));
    p.trace[28672 + 4 * blockIdx.x] = (long long)g;
    p.trace[28672 + 4 * blockIdx.x + 2] = clock64();
  }
#endif
  {  // token table: static tokens, then unclaimed tail entries
    const int sstat = max(0, p.s_q / ncl - TAIL) * ncl;
    const int ns = cl < sstat ? (sstat - 1 - cl) / ncl + 1 : 0;
    if (threadIdx.x < NTSEQ)
      ((volatile int*)(smem + OFF_TSEQ))[threadIdx.x] = (int)threadIdx.x < ns ? cl + (int)threadIdx.x * ncl : -1;
    if (threadIdx.x == 0) ((volatile int*)(smem + OFF_TPTR))[1] = 0;
  }
  if (threadIdx.x == 0) {
    if (sb & 1023) __trap();
    for (int s = 0; s < NS; ++s) {
      px::bar_init(bar(FULL + s), 2 + 64);  // two validity words + the 64 gather threads' cp.async arrivals
      px::bar_init(bar(PEERFULL + s), 1);
      px::bar_init(bar(EMPTY + s), 1);
    }
    px::bar_init(bar(SFULL), 1);
    px::bar_init(bar(SFREE), 8);
    px::bar_init(bar(PFULL), 8);
    px::bar_init(bar(PVDONE), 1);
    px::bar_init(bar(QKLAST), 1);
    px::bar_init(bar(QREADY), 16);
    px::bar_init(bar(OFULL), 1);
    px::bar_init(bar(OFREE), 16);
    px::bar_init_fence();
  }
  if (warp == 0) {
    px::tm_alloc2(sb + OFF_TPTR, 512);
    px::tm_relinquish2();
  }
  px::tc_before();
  px::cluster_arrive_wait();
  px::tc_after();
  const uint32_t tmem = *(volatile uint32_t*)(smem + OFF_TPTR);

  if (warp == 5) {
    px::regs_dec<32>();
    if (rank == 0) {
      pv_role(p, sb, tmem, cl, ncl);
    } else if (lane == 0) {
      // tail claimer: entry k is claimed once the token warps work on token k - 2; published to both CTAs
      const int sstat = max(0, p.s_q / ncl - TAIL) * ncl;
      const int ns = cl < sstat ? (sstat - 1 - cl) / ncl + 1 : 0;
      volatile int* prog = (volatile int*)(smem + OFF_TPTR + 4);
      for (int k = ns; k < NTSEQ; ++k) {
        while (*prog < k - 2) __nanosleep(1000);
        // (the table's last entry only ends it: a token claimed there would be lost)
        int t = k == NTSEQ - 1 ? p.s_q : sstat + atomicAdd(&g_tail[0], 1);
        if (t >= p.s_q) {
          t = p.s_q;
          if (atomicAdd(&g_tail[1], 1) == ncl - 1) {  // every pair is done claiming: reset for the next launch
            atomicExch(&g_tail[0], 0);
            atomicExch(&g_tail[1], 0);
          }
        }
        const uint32_t a = sb + OFF_TSEQ + 4 * k;
        asm volatile("st.volatile.shared.b32 [%0], %1;" ::"r"(a), "r"(t) : "memory");
        asm volatile("st.shared::cluster.b32 [%0], %1;" ::"r"(px::peer_addr(a, 0)), "r"(t) : "memory");
        if (t >= p.s_q) break;
      }
    }
  } else if (warp >= 12) {
    px::regs_inc<112>();
    token_role(p, smem, sb, tmem, rank, cl, ncl, warp & 3, (warp - 12) >> 2, lane);
  } else if (warp >= 8) {
    // pool: 20 warps x 96 = 1920 >= 8 x 32 (gather, issuers) + 4 x 184 + 8 x 112 = 1888
    px::regs_inc<184>();
    softmax_role(p, smem);
  } else if (warp == 4) {
    px::regs_dec<32>();
    if (rank == 0) {
      qk_role(p, sb, tmem, cl, ncl);
    } else if (lane == 0) {
      // relay: this CTA's tile-full -> the even CTA's PEERFULL, in tile order
      uint32_t G = 0;
      int k = 0;
      for (int t = tseq(sb, 0); t < p.s_q; t = tseq(sb, ++k)) {
        const int ng = ntiles(p, t);
        for (int g = 0; g < ng; ++g, ++G) {
          px::bar_wait(bar(FULL + G % NS), (G / NS) & 1);
          px::fence_async_smem();
          px::bar_arrive_cl_relaxed(bar0(PEERFULL + G % NS));
        }
      }
    }
  } else {
    px::regs_dec<32>();
    // ------------------------------------------------------------------------------------ gather (cp.async)
    // A warp's cp.async completion arrive (noinc) holds the warp until its copies have landed, so in-flight tiles =
    // gather warps working on different data: warp w copies key half w & 1 (32 keys) of the units with
    // U % 2 == w >> 1. A warp instruction copies one key: 32 lanes x 16 B = this CTA's 512-B half row, written in the
    // SW128 layout [64-dim block][key][128 B]. Each warp also publishes the validity word of its half.
    const uint64_t pol = px::pol_evict_last();
    // six gather warps (0..3, 6, 7): pair up takes the tiles G % 3 == up
    const int gw = warp < 4 ? warp : warp - 2, hk = gw & 1, up = gw >> 1;
    const char* kvb = p.kv + 512 * rank + 16 * (lane & 7);
    volatile uint32_t* mask = (volatile uint32_t*)(smem + OFF_MASK);
    int k = 0, t = tseq(sb, 0), g = 0, ng = t < p.s_q ? ntiles(p, t) : 0;
    auto adv = [&]() {
      if (++g == ng) {
        g = 0;
        t = tseq(sb, ++k);
        ng = t < p.s_q ? ntiles(p, t) : 0;
      }
    };
    uint32_t r0 = 0, m0 = 0;  // byte offset of the row of key 32hk + lane (s_kv * row < 4 GB, host-checked)
    auto load = [&]() {
      const int L = topk_len(p, t);
      const int k0 = TK * g + 32 * hk + lane;
      const int v0 = p.idx[(size_t)t * p.idx_stride + k0];
      const bool ok0 = k0 < L && v0 >= 0 && v0 < p.s_kv;
      m0 = __ballot_sync(0xffffffffu, ok0);
      r0 = ok0 ? (uint32_t)v0 * (uint32_t)p.kv_row : 0u;
    };
    uint32_t G = 0;
    for (int j = 0; j < up; ++j) {
      adv();
      ++G;
    }
    if (t < p.s_q) load();
    while (t < p.s_q) {
      const uint32_t cm0 = m0, Gc = G;
      // this tile's row offsets (8 per lane), before the slot wait
      const char* o[8];
#pragma unroll
      for (int ii = 0; ii < 8; ++ii) o[ii] = kvb + __shfl_sync(0xffffffffu, r0, 4 * ii + (lane >> 3));
      adv();
      adv();
      adv();
      G += 3;
      if (t < p.s_q) load();  // next unit of this warp (latency overlaps the slot wait)
      const int s = Gc % NS;
      if (Gc >= NS) px::bar_wait(bar(EMPTY + s), ((Gc / NS) - 1) & 1);
      if (lane == 0) {
        if (hk == 0) TR(7, Gc);
        mask[2 * s + hk] = cm0;
        px::bar_arrive(bar(FULL + s));
      }
      // an instruction copies 128 B (one 64-dim block) of four keys: lane l takes key 4i + l / 8 of the warp's 32,
      // 16-B piece l % 8 of the block (8 row offsets per lane instead of 32 broadcasts)
      // key k = 32hk + 4ii + lane / 8: k % 8 = 4 (ii % 2) + lane / 8, so the swizzled piece depends on ii's parity only
      const uint32_t dst = sb + OFF_RING + s * SLOT + (32 * hk + (lane >> 3)) * 128;
      const uint32_t d0 = dst + ((((lane & 7) ^ (lane >> 3)) & 7) << 4), d1 = dst + ((((lane & 7) ^ (4 + (lane >> 3))) & 7) << 4);
      if (!(p.dbg & 1)) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
#pragma unroll
          for (int ii = 0; ii < 8; ++ii)
            px::cp_async16((ii & 1 ? d1 : d0) + b * 8192 + ii * 512, o[ii] + 128 * b, pol);
        }
      }
      px::cp_async_arrive(bar(FULL + s));
    }
  }

  __syncwarp();
  px::tc_before();
  px::cluster_arrive_wait();
  if (warp == 0) px::tm_dealloc2(tmem, 512);
#ifdef DSA_TRACE
  if (p.trace && threadIdx.x == 0) {
    uint64_t g;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(g));
    p.trace[28672 + 4 * blockIdx.x + 1] = (long long)g;
    p.trace[28672 + 4 * blockIdx.x + 3] = clock64();
  }
#endif
}

}  // namespace dsa

// ---------------------------------------------------------------------------------------------------- host
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void launch(torch::Tensor q, torch::Tensor kv, torch::Tensor idx, torch::Tensor pos, torch::Tensor cs,
            c10::optional<torch::Tensor> sink, c10::optional<torch::Tensor> tlen, double scale, torch::Tensor out,
            torch::Tensor sf, torch::Tensor ml, torch::Tensor lse, int64_t ncl, int64_t dbg,
            c10::optional<torch::Tensor> trace) {
  using namespace dsa;
  TORCH_CHECK(q.is_contiguous() && q.size(1) == HQ && q.size(2) == 512);
  TORCH_CHECK(kv.stride(2) == 1 && idx.stride(2) == 1 && idx.size(2) == TOPK);
  TORCH_CHECK((kv.stride(0) * 2) % 16 == 0 && ((uintptr_t)kv.data_ptr() & 15) == 0);
  TORCH_CHECK((uint64_t)kv.size(0) * (uint64_t)kv.stride(0) * 2 < (1ull << 32), "kv rows must fit 32-bit byte offsets");
  const int s_q = q.size(0), s_kv = kv.size(0);
  TORCH_CHECK(s_q / ncl + 8 <= NTSEQ, "token table too small for s_q / pairs");
  Params p;
  p.q = (const uint4*)q.data_ptr();
  p.kv = (const char*)kv.data_ptr();
  p.kv_row = kv.stride(0) * 2;
  p.idx = idx.data_ptr<int>();
  p.idx_stride = idx.stride(0);
  p.pos = pos.data_ptr<int>();
  p.cs = cs.data_ptr<float>();
  p.sink = sink.has_value() ? sink->data_ptr<float>() : nullptr;
  p.tlen = tlen.has_value() ? tlen->data_ptr<int>() : nullptr;
  p.out = (uint8_t*)out.data_ptr();
  p.sf = sf.data_ptr<int>();
  p.ml = ml.data_ptr<float>();
  p.lse = lse.data_ptr<float>();
  p.s_q = s_q;
  p.s_kv = s_kv;
  p.scale = (float)scale;
  p.dbg = (int)dbg;
  p.trace = trace.has_value() ? (long long*)trace->data_ptr() : nullptr;
  static bool init = false;
  if (!init) {
    TORCH_CHECK(cudaFuncSetAttribute(dsa_prefill, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM) == cudaSuccess);
    init = true;
  }
  dsa_prefill<<<2 * ncl, NTHR, SMEM, at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("launch", &launch); }
""",
    'dsa_ptx.cuh': r"""#ifdef DSA_WD
#include <cstdio>
#endif
// Minimal sm_100a PTX wrappers used by dsa.cu (mbarrier, cluster, TMA gather4, tcgen05, misc numerics).
#pragma once
#include <cstdint>

#define DI __device__ __forceinline__

namespace px {

DI uint32_t saddr(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

// ------------------------------------------------------------------ cluster
// cluster index along x of a pair launch, read fresh (keeps role code from reloading spilled prologue values)
DI int pair_index() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%ctaid.x;" : "=r"(r));
  return (int)(r >> 1);
}
DI int pair_count() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%nctaid.x;" : "=r"(r));
  return (int)(r >> 1);
}
DI uint32_t cta_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
DI void cluster_arrive_wait() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
// address of the same shared variable in CTA `rank` of the cluster
DI uint32_t peer_addr(uint32_t a, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
  return r;
}

// ------------------------------------------------------------------ mbarrier
DI void bar_init(uint32_t b, uint32_t n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(b), "r"(n)); }
DI void bar_init_fence() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
DI void bar_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(b) : "memory"); }
// arrive on a barrier given by a shared::cluster address (local or peer)
DI void bar_arrive_cl(uint32_t b) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(b) : "memory");
}
DI void bar_arrive_cl_relaxed(uint32_t b) {
  asm volatile("mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0];" ::"r"(b) : "memory");
}
DI void bar_expect_tx_cl(uint32_t b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cluster.shared::cluster.b64 _, [%0], %1;" ::"r"(b), "r"(bytes) : "memory");
}
#if defined(DSA_WD)
// watchdog (debug builds): report a wait that does not complete within ~2 s, then trap
DI void bar_wait(uint32_t b, uint32_t parity) {
  const long long t0 = clock64();
  while (true) {
    uint32_t r;
    asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, 100000;\n\tselp.u32 %0, 1, 0, p;\n}"
                 : "=r"(r) : "r"(b), "r"(parity) : "memory");
    if (r) return;
    const long long dt = clock64() - t0;
    if (dt > 2000000000ll && dt < 2000000000ll + 4000000ll && (threadIdx.x & 31) == 0)
      printf("WD block %d warp %d bar 0x%x parity %u\n", (int)blockIdx.x, (int)(threadIdx.x >> 5), b & 0xffff, parity);
    if (dt > 8000000000ll) __trap();
  }
}
#elif defined(DSA_SPIN)
DI void bar_wait(uint32_t b, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred p;\n"
      "W_%=:\n\tmbarrier.test_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "\t@!p bra W_%=;\n}" ::"r"(b), "r"(parity)
      : "memory");
}
#else
DI void bar_wait(uint32_t b, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred p;\n"
      "W_%=:\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, 10000000;\n"
      "\t@!p bra W_%=;\n}" ::"r"(b), "r"(parity)
      : "memory");
}
#endif
// wait by polling with timed sleeps (no wake-ups on unrelated barrier traffic): for long waits off the critical path
DI void bar_wait_sleep(uint32_t b, uint32_t parity, uint32_t ns) {
  asm volatile(
      "{\n\t.reg .pred p;\n"
      "W_%=:\n\tmbarrier.test_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "\t@p bra D_%=;\n"
      "\tnanosleep.u32 %2;\n"
      "\tbra W_%=;\n"
      "D_%=:\n}" ::"r"(b), "r"(parity), "r"(ns)
      : "memory");
}
// wait with cluster-scope acquire (for barriers that receive arrivals from the peer CTA)
DI void bar_wait_cl(uint32_t b, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred p;\n"
      "W_%=:\n\tmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1, 10000000;\n"
      "\t@!p bra W_%=;\n}" ::"r"(b), "r"(parity)
      : "memory");
}

// non-blocking probe: has the phase with this parity completed?
DI bool bar_test(uint32_t b, uint32_t parity) {
  uint32_t r;
  asm volatile(
      "{\n\t.reg .pred p;\n\tmbarrier.test_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n}"
      : "=r"(r) : "r"(b), "r"(parity) : "memory");
  return r != 0;
}
DI bool bar_test_cl(uint32_t b, uint32_t parity) {
  uint32_t r;
  asm volatile(
      "{\n\t.reg .pred p;\n\tmbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n}"
      : "=r"(r) : "r"(b), "r"(parity) : "memory");
  return r != 0;
}

// ------------------------------------------------------------------ TMA
DI uint64_t pol_evict_last() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}
DI uint64_t pol_evict_first() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
// 4 rows x box-width columns of a 2-D tensor map -> 4 consecutive (swizzled) rows at dst; completion bytes go to
// `bar` (shared::cluster address, may be the peer CTA's barrier)
DI void gather4(uint32_t dst, const void* tmap, int col, int r0, int r1, int r2, int r3, uint32_t bar, uint64_t pol) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
      " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;" ::"r"(dst),
      "l"(tmap), "r"(col), "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(bar), "l"(pol)
      : "memory");
}
// 16-B global -> shared copy (L2 cache hint); completion tracked by cp_async_arrive
DI void cp_async16(uint32_t dst, const void* src, uint64_t pol) {
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "l"(pol) : "memory");
}
// arrive on a local barrier when all prior cp.async of this thread completed (no pending-count increment)
DI void cp_async_arrive(uint32_t b) { asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(b) : "memory"); }
DI void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> DI void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }
DI void prefetch_tmap(const void* tmap) { asm volatile("prefetch.tensormap [%0];" ::"l"(tmap) : "memory"); }
DI void l2_prefetch(const void* p, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes) : "memory");
}

// ------------------------------------------------------------------ global memory with cache hints
DI uint4 ld_stream(const void* p, uint64_t pol) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p), "l"(pol));
  return v;
}
DI void st_stream(void* p, uint4 v, uint64_t pol) {
  asm volatile("st.global.L1::no_allocate.L2::cache_hint.v4.u32 [%0], {%1, %2, %3, %4}, %5;" ::"l"(p), "r"(v.x), "r"(v.y),
               "r"(v.z), "r"(v.w), "l"(pol)
               : "memory");
}

// 32-B store (sm_100 256-bit vector access)
DI void st_stream32(void* p, const uint32_t* w, uint64_t pol) {
  asm volatile("st.global.L1::no_allocate.L2::cache_hint.v8.u32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8}, %9;" ::"l"(p),
               "r"(w[0]), "r"(w[1]), "r"(w[2]), "r"(w[3]), "r"(w[4]), "r"(w[5]), "r"(w[6]), "r"(w[7]), "l"(pol)
               : "memory");
}

// ------------------------------------------------------------------ tcgen05
DI void tm_alloc2(uint32_t dst_smem, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(dst_smem), "r"(ncols));
}
DI void tm_relinquish2() { asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;"); }
DI void tm_dealloc2(uint32_t t, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::"r"(t), "r"(ncols));
}
DI void tc_before() { asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
DI void tc_after() { asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }

// D[tmem] (+)= A * B, pair of CTAs (issued by the even CTA)
DI void mma2_ss(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t acc) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n"
      "\ttcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}" ::"r"(d),
      "l"(a), "l"(b), "r"(idesc), "r"(acc));
}
// same, A kept in the collector for the next MMA (fill) / taken from it and released (lastuse)
DI void mma2_ss_afill(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t acc) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n"
      "\ttcgen05.mma.cta_group::2.kind::f16.collector::a::fill [%0], %1, %2, %3, p;\n}" ::"r"(d),
      "l"(a), "l"(b), "r"(idesc), "r"(acc));
}
DI void mma2_ss_alast(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc, uint32_t acc) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n"
      "\ttcgen05.mma.cta_group::2.kind::f16.collector::a::lastuse [%0], %1, %2, %3, p;\n}" ::"r"(d),
      "l"(a), "l"(b), "r"(idesc), "r"(acc));
}
DI void mma2_ts(uint32_t d, uint32_t a, uint64_t b, uint32_t idesc, uint32_t acc) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n"
      "\ttcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, p;\n}" ::"r"(d),
      "r"(a), "l"(b), "r"(idesc), "r"(acc));
}
// arrive (once) on the barrier at offset `bar` in both CTAs of the pair when all prior MMAs completed
DI void mma2_commit(uint32_t bar) {
  asm volatile(
      "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" ::"r"(bar),
      "h"((uint16_t)3)
      : "memory");
}

// bf16 x bf16 -> fp32 instruction descriptor
constexpr uint32_t idesc_bf16(int M, int N, int a_mn, int b_mn) {
  return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)a_mn << 15) | ((uint32_t)b_mn << 16) |
         ((uint32_t)(N >> 3) << 17) | ((uint32_t)(M >> 4) << 24);
}
// shared-memory matrix descriptor, 128-byte swizzle
DI uint64_t sdesc_sw128(uint32_t addr, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)(lbo >> 4) << 16) | ((uint64_t)(sbo >> 4) << 32) |
         (1ull << 46) | (2ull << 61);
}

// shared-memory matrix descriptor, no swizzle (8-row x 16-B core matrices; lbo: K direction, sbo: M/N direction)
DI uint64_t sdesc_none(uint32_t addr, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)(lbo >> 4) << 16) | ((uint64_t)(sbo >> 4) << 32) | (1ull << 46);
}

DI void tm_ld64(uint32_t a, uint32_t* r) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x64.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
      "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, [%64];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
        "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),
        "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),
        "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]),
        "=r"(r[32]), "=r"(r[33]), "=r"(r[34]), "=r"(r[35]), "=r"(r[36]), "=r"(r[37]), "=r"(r[38]), "=r"(r[39]),
        "=r"(r[40]), "=r"(r[41]), "=r"(r[42]), "=r"(r[43]), "=r"(r[44]), "=r"(r[45]), "=r"(r[46]), "=r"(r[47]),
        "=r"(r[48]), "=r"(r[49]), "=r"(r[50]), "=r"(r[51]), "=r"(r[52]), "=r"(r[53]), "=r"(r[54]), "=r"(r[55]),
        "=r"(r[56]), "=r"(r[57]), "=r"(r[58]), "=r"(r[59]), "=r"(r[60]), "=r"(r[61]), "=r"(r[62]), "=r"(r[63])
      : "r"(a));
}
DI void tm_ld32(uint32_t a, uint32_t* r) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
        "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),
        "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),
        "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
      : "r"(a));
}
DI void tm_st32(uint32_t a, const uint32_t* r) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x32.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,"
      "%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32};" ::"r"(a),
      "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]), "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]), "r"(r[8]), "r"(r[9]),
      "r"(r[10]), "r"(r[11]), "r"(r[12]), "r"(r[13]), "r"(r[14]), "r"(r[15]), "r"(r[16]), "r"(r[17]), "r"(r[18]),
      "r"(r[19]), "r"(r[20]), "r"(r[21]), "r"(r[22]), "r"(r[23]), "r"(r[24]), "r"(r[25]), "r"(r[26]), "r"(r[27]),
      "r"(r[28]), "r"(r[29]), "r"(r[30]), "r"(r[31]));
}
DI void tm_ld_wait() { asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
DI void tm_st_wait() { asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }

// ------------------------------------------------------------------ misc
DI void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
// named barrier that also ORs a predicate over the participating threads
DI bool nbar_or(int id, int n, bool v) {
  uint32_t r;
  asm volatile("{\n\t.reg .pred p, q;\n\tsetp.ne.u32 q, %3, 0;\n\tbarrier.cta.red.or.pred p, %1, %2, q;\n\tselp.u32 %0, 1, 0, p;\n}"
               : "=r"(r) : "r"(id), "r"(n), "r"((uint32_t)v) : "memory");
  return r != 0;
}
DI void nbar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory"); }
DI void nbar(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
template <int N> DI void regs_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(N)); }
template <int N> DI void regs_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(N)); }
DI bool elect() {
  uint32_t p;
  asm volatile("{\n\t.reg .pred P;\n\telect.sync _|P, 0xffffffff;\n\tselp.b32 %0, 1, 0, P;\n}" : "=r"(p));
  return p != 0;
}

DI float exp2a(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
DI float log2a(float x) {
  float y;
  asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
// paired fp32 ops
DI uint64_t pk2(float a, float b) {
  uint64_t r;
  asm("mov.b64 %0, {%1, %2};" : "=l"(r) : "f"(a), "f"(b));
  return r;
}
DI void up2(uint64_t v, float& a, float& b) { asm("mov.b64 {%0, %1}, %2;" : "=f"(a), "=f"(b) : "l"(v)); }
DI uint64_t fma2(uint64_t a, uint64_t b, uint64_t c) {
  uint64_t r;
  asm("fma.rn.f32x2 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c));
  return r;
}
DI uint64_t add2(uint64_t a, uint64_t b) {
  uint64_t r;
  asm("add.rn.f32x2 %0, %1, %2;" : "=l"(r) : "l"(a), "l"(b));
  return r;
}
DI uint64_t mul2(uint64_t a, uint64_t b) {
  uint64_t r;
  asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(r) : "l"(a), "l"(b));
  return r;
}
// (lo, hi) -> bf16x2 with lo in the low half
DI uint32_t bf16x2(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}
// (a0, a1, a2, a3) -> 4 e4m3 bytes, a0 in the low byte
DI uint32_t e4m3x4(float a0, float a1, float a2, float a3) {
  uint32_t r;
  asm("{\n\t.reg .b16 lo, hi;\n\tcvt.rn.satfinite.e4m3x2.f32 lo, %2, %1;\n\tcvt.rn.satfinite.e4m3x2.f32 hi, %4, %3;\n\t"
      "mov.b32 %0, {lo, hi};\n\t}" : "=r"(r) : "f"(a0), "f"(a1), "f"(a2), "f"(a3));
  return r;
}
// (lo, hi) -> e4m3x2 with lo in the low byte
DI uint32_t e4m3x2(float lo, float hi) {
  uint16_t r;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
  return r;
}

}  // namespace px
""",
    'kernel.py': r'''"""DeepSeek-V4.1 prefill: Q RoPE + sparse MQA attention with sink + conjugate O RoPE + FP8 cast (one kernel, sm_100a).

dsa.cu is JIT-compiled once on import (outside timing).
Q_PERM: per token, q is consumed in 16-byte pieces ordered [cta r][col half c][TMEM quadrant w][piece k][lane]:
piece k of lane l in quadrant w holds head 64r + (32w + l) % 64, dims 256*(w // 2) + 128c + 8k .. +8, so each warp
load instruction reads 512 contiguous bytes.
O_PERM: within a group of 8 heads the stored channel 256*b + 32*j + e is head j, dim 32*b + e (8 heads of one
32-dim block are 256 contiguous bytes); scale blocks are natural 32-dim blocks of one head.
"""
import hashlib
import os

import torch
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir("/usr/local/cuda"):
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0a"
_BUILD = os.path.join(os.path.expanduser("~"), ".cache", "dsa_sol_build")
os.makedirs(_BUILD, exist_ok=True)

_TRACE = os.environ.get("DSA_TRACE") == "1"
_SRC = [os.path.join(_HERE, f) for f in ("dsa.cu", "dsa_ptx.cuh")]
_TAG = hashlib.md5(b"".join(open(f, "rb").read() for f in _SRC)).hexdigest()[:10]
_mod = load(
    name="dsa_sol_" + _TAG + ("_tr" if _TRACE else "") + ("_spin" if os.environ.get("DSA_SPIN") == "1" else "") + ("_wd" if os.environ.get("DSA_WD") == "1" else ""),
    sources=[os.path.join(_HERE, "dsa.cu")],
    extra_include_paths=[_HERE],
    extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_100a,code=sm_100a", "-lineinfo"]
    + (["-DDSA_TRACE"] if _TRACE else []) + (["-DDSA_SPIN"] if os.environ.get("DSA_SPIN") == "1" else []) + (["-DDSA_WD"] if os.environ.get("DSA_WD") == "1" else []),
    extra_ldflags=["-lcuda"],
    build_directory=_BUILD,
    verbose=False,
)

Q_PERM = [(64 * r + (32 * w + l) % 64) * 512 + 256 * (w // 2) + 128 * c + 8 * k + e
          for r in range(2) for c in range(2) for w in range(4) for k in range(16) for l in range(32) for e in range(8)]
O_PERM = [512 * j + 32 * b + e for b in range(16) for j in range(8) for e in range(32)]

dbg = 0
trace = None  # debug: int64 cuda tensor [8192] (DSA_TRACE=1 builds)
_NCL = {}


def run(q, kv, indices, token_positions, cos_sin_cache, attn_sink, topk_length, sm_scale, rms_norm_eps, enable_q_norm):
    if enable_q_norm:
        raise NotImplementedError("Q RMSNorm is off for DeepSeek-V4.1")
    dev = q.device
    s_q = q.shape[0]
    if dev not in _NCL:
        _NCL[dev] = torch.cuda.get_device_properties(dev).multi_processor_count // 2
    out = torch.empty((s_q, 16, 4096), dtype=torch.float8_e4m3fn, device=dev)
    sf = torch.empty((16, 32, s_q), dtype=torch.int32, device=dev).permute(2, 0, 1)
    ml = torch.empty((s_q, 128), dtype=torch.float32, device=dev)
    lse = torch.empty((s_q, 128), dtype=torch.float32, device=dev)
    _mod.launch(q, kv, indices, token_positions, cos_sin_cache, attn_sink, topk_length, float(sm_scale),
                out, sf, ml, lse, _NCL[dev], dbg, trace)
    return out, sf, ml, lse
''',
}



def run(*args):
    """Not a submission: this artifact is reference material (see `_SOURCES`)."""
    raise NotImplementedError(
        "reference bundle (kernel.py JIT-builds sibling CUDA sources from disk and pins a code-generation target; the mechanical one-file rule cannot rewrite this loader shape): adapt it first (`forge lib adapt`)")

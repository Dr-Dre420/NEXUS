// NEXUS formatting vocabulary. One definition per unit, used everywhere.

const INR = new Intl.NumberFormat('en-IN', {
  style: 'currency', currency: 'INR', maximumFractionDigits: 0,
});
const INR2 = new Intl.NumberFormat('en-IN', {
  style: 'currency', currency: 'INR', minimumFractionDigits: 2, maximumFractionDigits: 2,
});

export const currency = (v, precise = false) =>
  !Number.isFinite(v) ? '—' : (precise ? INR2 : INR).format(v);

/** A probability in [0,1], shown as a probability — never as a percentile. */
export const probability = (v, dp = 4) =>
  !Number.isFinite(v) ? '—' : v.toFixed(dp);

/** A probability rendered as a percent of 1. Use only where a rate is meant. */
export const percent = (v, dp = 2) =>
  !Number.isFinite(v) ? '—' : `${(v * 100).toFixed(dp)}%`;

/** A 0-100 rank within the frozen cohort. Distinct from probability. */
export const percentile = (v, dp = 1) =>
  !Number.isFinite(v) ? '—' : `${v.toFixed(dp)}`;

/** An unbounded index (e.g. a sum of exposures). Never suffixed with %. */
export const indexValue = (v, dp = 4) =>
  !Number.isFinite(v) ? '—' : v.toFixed(dp);

export const count = (v) =>
  !Number.isFinite(v) ? '—' : new Intl.NumberFormat('en-IN').format(v);

export const ordinalSuffix = (n) => {
  const s = ['th', 'st', 'nd', 'rd'], v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
};

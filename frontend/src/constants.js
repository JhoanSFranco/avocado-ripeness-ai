/** Ripeness stages (must match avocado_ai.domain.RipenessStage). */
export const STAGES = [
  { stage: 1, label: 'Unripe' },
  { stage: 2, label: 'Breaking' },
  { stage: 3, label: 'Ripe' },
  { stage: 4, label: 'Very ripe' },
  { stage: 5, label: 'Overripe' },
];

/** Colors per confidence level returned by the API. */
export const CONFIDENCE_COLORS = {
  high: '#2ecc71',
  medium: '#e6a817',
  low: '#e05252',
};

export const CONFIDENCE_LABELS = {
  high: 'High confidence',
  medium: 'Medium confidence',
  low: 'Low confidence',
};

export const MAX_HISTORY_ITEMS = 5;

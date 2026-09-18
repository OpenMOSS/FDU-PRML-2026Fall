"""Decision tree visualization utility.

Features:
 - Robust to arbitrary (non-contiguous / negative / string) class labels.
 - Safe color mapping.
 - Optional display of split score & leaf sample counts.
 - Works with DecisionTreeClassifier (expects node attributes: feat_idx, threshold,
   value, split_score, left, right, leaf_num).
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import matplotlib.pyplot as plt

try:  # Chinese font support (ignore if missing)
	plt.rcParams['font.sans-serif'] = ['SimHei']
except Exception:  # pragma: no cover
	pass

# Style presets
_NON_LEAF_STYLE = dict(boxstyle="round", facecolor="white", mutation_scale=1.2, ls="--")
_LEAF_STYLE = dict(boxstyle="square", mutation_scale=1.2)
_ARROW = dict(arrowstyle="<-")

_PALETTE = [
	'peachpuff', 'yellowgreen', 'palevioletred', 'skyblue', 'darkorange',
	'blueviolet', 'slategrey', 'khaki', 'silver', 'teal'
]


def _annotate(ax, text: str, center: Tuple[float, float], parent: Tuple[float, float], style: Dict[str, Any], font_size: float):
	ax.annotate(text,
			xy=parent, xycoords='axes fraction',
			xytext=center, textcoords='axes fraction',
			size=font_size, va="bottom", ha="center",
			bbox=style, arrowprops=_ARROW)


def _collect_labels(node, labels: Set[Any]):
	if node is None:
		return
	if node.value is not None:
		labels.add(node.value)
	else:
		_collect_labels(node.left, labels)
		_collect_labels(node.right, labels)


def _build_mappings(root, class_names: Optional[List[str]]):
	labels: Set[Any] = set()
	_collect_labels(root, labels)
	sorted_labels = sorted(list(labels), key=lambda v: (str(type(v)), v))
	# Class name mapping
	if class_names and len(class_names) == len(sorted_labels):
		class_map = {lab: class_names[i] for i, lab in enumerate(sorted_labels)}
	else:
		class_map = {lab: str(lab) for lab in sorted_labels}
	color_map = {lab: i for i, lab in enumerate(sorted_labels)}
	return class_map, color_map


def _format_node_strings(node, feat_names: List[str], class_map: Dict[Any, str],
						 show_split_score: bool, show_leaf_samples: bool):
	# Feature string
	if node.feat_idx is not None and 0 <= int(node.feat_idx) < len(feat_names):
		feat_str = f"feat: {feat_names[int(node.feat_idx)]}"
	else:
		feat_str = f"feat_id: {node.feat_idx}"
	# Threshold
	thr_str = f"threshold: {node.threshold if node.threshold is not None else '-'}"
	# Class label
	class_disp = class_map.get(node.value, str(node.value)) if node.value is not None else '-'
	class_str = f"class: {class_disp}"
	# Optional extras
	extra_lines = []
	if show_split_score and getattr(node, 'split_score', None) is not None and (node.left or node.right):
		extra_lines.append(f"gain: {node.split_score:.4f}")
	if show_leaf_samples and getattr(node, 'leaf_num', None) is not None and (node.left or node.right):
		extra_lines.append(f"leaves: {node.leaf_num}")
	return feat_str, thr_str, class_str, extra_lines


def _compute_positions(node, depth: int, info: Dict[str, Any]):
	"""Assign x positions (leaf order) and record depth/leaf count."""
	if node is None:
		return
	node._depth = depth  # type: ignore[attr-defined]
	if node.left is None and node.right is None:  # leaf
		node._x = info['next_x']  # type: ignore[attr-defined]
		info['next_x'] += 1
		info['total_leaves'] += 1
	else:
		_compute_positions(node.left, depth + 1, info)
		_compute_positions(node.right, depth + 1, info)
		# internal node x = mean(child x)
		child_x = []
		if node.left is not None:
			child_x.append(getattr(node.left, '_x'))
		if node.right is not None:
			child_x.append(getattr(node.right, '_x'))
		if child_x:
			node._x = sum(child_x) / len(child_x)  # type: ignore[attr-defined]
	info['max_depth'] = max(info['max_depth'], depth)


def _plot_subtree(clf,
			  node,
			  parent_pt,
			  ax,
			  feat_names,
			  class_map,
			  color_map,
			  show_split_score: bool,
			  show_leaf_samples: bool,
			  top_padding: float,
			  bottom_padding: float,
			  total_leaves: int,
			  max_depth: int,
			  base_font_size: float,
			  min_font_size: float,
			  depth_font_decay: float,
			  box_scale_internal: float,
			  box_scale_leaf: float):
	feat_str, thr_str, class_str, extras = _format_node_strings(
		node, feat_names, class_map, show_split_score, show_leaf_samples)

	# Normalized horizontal position
	if total_leaves > 1:
		x_norm = getattr(node, '_x') / (total_leaves - 1)
	else:
		x_norm = 0.5

	# Vertical mapping based on depth
	depth = getattr(node, '_depth', 0)
	if max_depth > 0:
		y_rel = 1.0 - depth / max_depth  # root near top
	else:
		y_rel = 1.0
	usable_h = max(1.0 - top_padding - bottom_padding, 1e-6)
	y_mapped = bottom_padding + y_rel * usable_h
	center_pt = (x_norm, y_mapped)
	if parent_pt == (0, 0):
		parent_pt = center_pt

	# Dynamic font size (smaller for deeper nodes)
	font_size = max(min_font_size, base_font_size - depth_font_decay * depth)

	if node.left is None and node.right is None:  # leaf
		label = "\n" + class_str + "\n"
		raw_label = node.value
		leaf_style = dict(_LEAF_STYLE)  # copy
		leaf_style['mutation_scale'] = box_scale_leaf * 10  # mutation_scale expects larger numbers
		if raw_label in color_map:
			leaf_style['fc'] = _PALETTE[color_map[raw_label] % len(_PALETTE)]
		else:  # fallback
			leaf_style['fc'] = _PALETTE[hash(raw_label) % len(_PALETTE)]
		_annotate(ax, label, center_pt, parent_pt, leaf_style, font_size)
	else:  # internal node
		lines = [feat_str, thr_str] + extras + [class_str]
		inner_style = dict(_NON_LEAF_STYLE)
		inner_style['mutation_scale'] = box_scale_internal * 10
		_annotate(ax, "\n".join(lines), center_pt, parent_pt, inner_style, font_size)

	# Recurse
	if node.left is not None:
		_plot_subtree(clf, node.left, center_pt, ax, feat_names,
				  class_map, color_map, show_split_score, show_leaf_samples,
				  top_padding, bottom_padding, total_leaves, max_depth,
				  base_font_size, min_font_size, depth_font_decay,
				  box_scale_internal, box_scale_leaf)
	if node.right is not None:
		_plot_subtree(clf, node.right, center_pt, ax, feat_names,
				  class_map, color_map, show_split_score, show_leaf_samples,
				  top_padding, bottom_padding, total_leaves, max_depth,
				  base_font_size, min_font_size, depth_font_decay,
				  box_scale_internal, box_scale_leaf)


def plot_tree(clf,
		  feat_names: Optional[List[str]] = None,
		  class_names: Optional[List[str]] = None,
		  show_split_score: bool = True,
		  show_leaf_samples: bool = False,
		  figsize: Tuple[int, int] = (18, 18),
		  top_padding: float = 0.15,
		  bottom_padding: float = 0.05,
		  base_font_size: float = 10.0,
		  min_font_size: float = 6.0,
		  depth_font_decay: float = 0.6,
		  box_scale_internal: float = 0.6,
		  box_scale_leaf: float = 0.7):
	"""Plot a decision tree.

	Parameters
	----------
	clf : trained DecisionTreeClassifier (with .root, .tree_leaf_num, .tree_depth)
	feat_names : list[str], optional feature names (len must match n_features)
	class_names : list[str], optional class display names (len must match #labels in tree)
	show_split_score : bool, show each internal node's split_score (gain)
	show_leaf_samples : bool, show number of leaves under an internal node
	figsize : figure size
	"""
	feat_names = feat_names or []
	fig = plt.figure(figsize=figsize, facecolor='white')
	ax = plt.subplot(111, frameon=False, xticks=[], yticks=[])

	if getattr(clf, 'root', None) is None:
		ax.text(0.5, 0.5, 'Empty tree', ha='center', va='center')
		return fig, ax

	class_map, color_map = _build_mappings(clf.root, class_names)

	# Pre-compute positions for a clean, centered layout
	info = {'next_x': 0, 'total_leaves': 0, 'max_depth': 0}
	_compute_positions(clf.root, 0, info)
	total_leaves = max(info['total_leaves'], 1)
	max_depth = max(info['max_depth'], 1)

	_plot_subtree(clf, clf.root, (0, 0), ax, feat_names,
			  class_map, color_map, show_split_score, show_leaf_samples,
			  top_padding, bottom_padding, total_leaves, max_depth,
			  base_font_size, min_font_size, depth_font_decay,
			  box_scale_internal, box_scale_leaf)

	ax.set_xlim(-0.05, 1.05)
	ax.set_ylim(0 - 1e-3, 1)
	return fig, ax


def attach_plot_method(cls):  # optional helper
	"""Monkey-patch a .plot() method into the classifier class (optional)."""
	def _plot(self, feat_names=None, class_names=None, **kwargs):  # noqa: D401
		return plot_tree(self, feat_names, class_names, **kwargs)
	if not hasattr(cls, 'plot'):
		setattr(cls, 'plot', _plot)
	return cls
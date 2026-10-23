#!/usr/bin/env python3
'''
Build a Mandyoc interfaces.txt from an SVG drawing (e.g. made in Inkscape).

Each filled shape is a body of rock described by its object properties
(Inkscape: Object > Object Properties):

    Label        name of the unit (layer_label). Falls back to the id.
    Title        optional, used in the legend of the plot.
    Description  first line: rheology (wet olivine, dry olivine, wet quartz,
                 dry quartz, basalt, plagioclase, air); next lines: key = value
                 with the layer properties, e.g.

                     wet quartz
                     C = 1.0
                     rho = 2800.0
                     H = 0.19e-6/2800.0

                 Keys: C, rho, H (required), and A, n, Q, V (optional, they
                 override the values of the rheology). Long names such as
                 density or radiogenic_heat_production are also accepted.

                 Optional: h = 120 km (or h = 120e3 m), the reference
                 thickness of the unit, so the drawing does not need to be in
                 vertical scale. See rescale_thickness:
                   - layers (units that reach both sides of the domain): their most
                     common drawn thickness becomes h and the same factor is
                     applied in every column, so thinned regions keep their
                     drawn proportion. Layers without h keep the drawn thickness;
                   - bodies (seeds, basins, weak zones, ...): without h they
                     follow the scaling of the unit they are drawn over; with
                     h they get that thickness and the unit around them
                     absorbs the difference, so the units below do not move;
                   - the bottom unit (e.g. asthenosphere) fills the rest of
                     the domain, so it can not have h.

Shapes with the same label are the same unit, so the description only needs
to be written once. A shape without description inherits the one of the group
(e.g. an Inkscape layer) that contains it. Shapes without any description are
ignored (annotations, arrows, ...). The fill colour is used only in the plot.

Domain: a shape labelled "domain" (usually a rectangle without fill, e.g. in its
own Inkscape layer) gives the model domain. Its bounding box is mapped to the
domain, top-left corner = (x=0, z=0), bottom-right = (Lx, -Lz), and its
description gives the dimensions and the resolution of the Mandyoc grid:

                     Lx = 1600 km
                     Lz = 400 km
                     dx = 2 km        (or Nx = 801)
                     dz = 2 km        (or Nz = 201)

It may also give the settings of the numerical experiment, which are returned
as text in domain["experiment"] and read by the generate_input_files*_svg.py
scripts: regime (rifting, compression, accordion, quiescence), velocity
(e.g. 1 cm/yr), moving_boundary (left or right), time_max (e.g. 100 Myr),
quiescence_time (e.g. 300 Myr), magmatism (on or off), magmatism_extraction (on or off)
magmatism_Ts0 (e.g. 1080, in oC) and preset_initial_temperature (folder or .nc file of a
preliminary scenario whose final thermal state is used as initial temperature).

Without a domain shape the SVG page is the domain and Lx, Lz, Nx must be given.
x and z are scaled independently, so the drawing may be vertically
exaggerated. Shapes are painted in document order (later shapes cover earlier
ones), as Inkscape renders them.

Since Mandyoc layers are bounded by single-valued interfaces z(x), the script
reads, for every column x, the sequence of units from bottom to top and finds a
single ordered stack containing all of them (shortest common supersequence).
Units absent in a column get zero thickness there. A seed inside the lower
crust becomes [..., lower crust, seed, lower crust, ...]. Each slot of the stack
is a MandyocLayer whose interface is its base depth, as in generate_input_files*.py.

Usage
-----
    python svg2interfaces.py scenario.svg -o interfaces.txt --plot setup.png
    python svg2interfaces.py scenario.svg --dx 1     (override the horizontal resolution of the SVG)

or, from another script:

    from svg2interfaces import read_domain, layers_from_svg
    domain = read_domain("scenario.svg")
    layers = layers_from_svg("scenario.svg", domain["Lx"], domain["Lz"], domain["Nx"])

Requires: numpy, matplotlib, svgelements (pip install svgelements)
'''

import argparse
import ast
import copy
import operator
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import Patch
import svgelements as se

from mandyoc_layers import MandyocLayer, get_rheology

SVG_NS = "{http://www.w3.org/2000/svg}"
INKSCAPE_LABEL = "{http://www.inkscape.org/namespaces/inkscape}label"
SHAPE_TAGS = {"rect", "path", "circle", "ellipse", "polygon", "polyline"}
DOMAIN_LABEL = "domain"  # label (or id) of the shape that gives the model domain
EXPERIMENT_KEYS = ("regime", "velocity", "moving_boundary", "time_max", "quiescence_time",
                   "magmatism", "magmatism_extraction", "magmatism_Ts0", "preset_initial_temperature")  # settings of the experiment in the domain description

# key in the description -> attribute of MandyocLayer
PROPERTY_KEYS = {
    "C": "effective_viscosity_scale_factor",
    "rho": "density",
    "H": "radiogenic_heat_production",
    "A": "pre_exponential_constant",
    "n": "power_law_exponent",
    "Q": "activation_energy",
    "V": "activation_volume",
}
PROPERTY_KEYS.update({attr: attr for attr in list(PROPERTY_KEYS.values())})

#Build layer_properties according to the order of the stack_layers
rheological_symbols = ['C', 'rho', 'H', 'A', 'n', 'Q', 'V']
rheological_properties = ['effective_viscosity_scale_factor',
                          'density',
                          'radiogenic_heat_production',
                          'pre_exponential_constant',
                          'power_law_exponent',
                          'activation_energy',
                          'activation_volume']

###############################################################################
# Functions
###############################################################################

_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}

def eval_number(expr):
    '''Evaluate a numeric expression such as "1.67e-6 / 2700.0" (only numbers and + - * / **).'''
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](_eval(node.operand))
        raise ValueError(f"invalid number: {expr!r}")
    return _eval(ast.parse(expr.strip(), mode="eval"))


def parse_thickness(value, where, name="h"):
    '''Read a length such as "120 km" or "120e3 m" and return it in m.'''
    match = re.fullmatch(r"(.+?)\s*(km|m)", value.strip())
    if match is None:
        sys.exit(f"{where}: give the unit of {name} (km or m), e.g. {name} = 120 km")
    try:
        number = eval_number(match.group(1))
    except (ValueError, SyntaxError):
        sys.exit(f"{where}: could not read the value of {name} = {value}")
    thickness = number * (1.0e3 if match.group(2) == "km" else 1.0)
    if thickness <= 0:
        sys.exit(f"{where}: {name} must be positive.")
    return thickness


def points_from_spacing(length, spacing, name):
    '''Number of grid points for a grid spacing; the spacing must divide the length.'''
    intervals = length / spacing
    if abs(intervals - round(intervals)) > 1e-6:
        sys.exit(f"{name} = {spacing / 1e3:g} km does not divide the domain ({length / 1e3:g} km).")
    return int(round(intervals)) + 1


def read_domain(svg_file):
    '''
    Read the shape labelled "domain" (see the module docstring).
    Return dict(id, Lx, Lz, Nx, Nz, experiment) with lengths in m, or None if the SVG has no domain shape.
    experiment is a dict with the settings of the numerical experiment given in the description (as text).
    '''
    root = ET.parse(svg_file).getroot()
    shapes = [el for el in root.iter() if el.tag.replace(SVG_NS, "") in SHAPE_TAGS
              and DOMAIN_LABEL in (el.get(INKSCAPE_LABEL), el.get("id"))]
    if not shapes:
        return None
    if len(shapes) > 1:
        sys.exit(f"The SVG has {len(shapes)} shapes labelled '{DOMAIN_LABEL}'; keep only one.")
    el = shapes[0]
    where = f"Domain shape (id={el.get('id')})"
    if el.get("id") is None:
        sys.exit(f"{where}: the domain shape has no id. Save the file with Inkscape to create it.")
    desc = el.find(SVG_NS + "desc")
    if desc is None or not (desc.text or "").strip():
        sys.exit(f"{where}: write Lx, Lz and the resolution (dx and dz, or Nx and Nz) in its description.")

    values, experiment = {}, {}
    for line in desc.text.splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        if "=" not in line:
            sys.exit(f"{where}: unexpected line '{line}' in the description.")
        key, value = (s.strip() for s in line.split("=", 1))
        if key in ("Lx", "Lz", "dx", "dz"):
            values[key] = parse_thickness(value, where, key)
        elif key in ("Nx", "Nz"):
            try:
                values[key] = int(value)
            except ValueError:
                sys.exit(f"{where}: {key} must be an integer.")
        elif key in EXPERIMENT_KEYS:
            experiment[key] = value
        else:
            sys.exit(f"{where}: unknown key '{key}'. Use Lx, Lz, dx, dz (or Nx, Nz) and {', '.join(EXPERIMENT_KEYS)}.")

    for key in ("Lx", "Lz"):
        if key not in values:
            sys.exit(f"{where}: missing {key} in the description.")
    domain = dict(id=el.get("id"), Lx=values["Lx"], Lz=values["Lz"], experiment=experiment)
    for d, N, L in (("dx", "Nx", "Lx"), ("dz", "Nz", "Lz")):
        if d in values:
            domain[N] = points_from_spacing(values[L], values[d], d)
            if N in values and values[N] != domain[N]:
                sys.exit(f"{where}: {d} and {N} do not agree; give only one of them.")
        elif N in values:
            domain[N] = values[N]
        else:
            sys.exit(f"{where}: give the resolution {d} (or {N}) in the description.")
    return domain


def parse_description(text, where):
    '''
    Parse the description of a shape into
    (rheology class, {MandyocLayer attribute: value}, reference thickness in m or None).
    '''
    rheology, props, thickness = None, {}, None
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        if "=" not in line:
            if rheology is not None:
                sys.exit(f"{where}: unexpected line '{line}' in the description.")
            rheology = line
            continue
        key, value = (s.strip() for s in line.split("=", 1))
        if key.lower() == "rheology":
            rheology = value
            continue
        if key == "h":
            thickness = parse_thickness(value, where)
            continue
        if key not in PROPERTY_KEYS:
            sys.exit(f"{where}: unknown property '{key}'. Use {', '.join(rheological_symbols)}.")
        try:
            props[PROPERTY_KEYS[key]] = eval_number(value)
        except (ValueError, SyntaxError):
            sys.exit(f"{where}: could not read the value of {key} = {value}")

    if rheology is None:
        sys.exit(f"{where}: the first line of the description must be the rheology (e.g. wet quartz).")
    try:
        rheology = get_rheology(rheology)
    except ValueError as err:
        sys.exit(f"{where}: {err}")
    missing = [s for s in ["C", "rho", "H"] if PROPERTY_KEYS[s] not in props]
    if missing:
        sys.exit(f"{where}: missing {', '.join(missing)} in the description.")
    return rheology, props, thickness


def read_units(svg_file):
    '''
    Read label, title and description of the shapes.
    Return {shape id: unit label} and {unit label: dict(rheology, props, title)}.
    '''
    root = ET.parse(svg_file).getroot()
    parent = {child: el for el in root.iter() for child in el}

    def child_text(el, tag):
        node = el.find(SVG_NS + tag)
        return node.text.strip() if node is not None and node.text and node.text.strip() else None

    def label_of(el):
        return el.get(INKSCAPE_LABEL) or el.get("id")

    shape_unit, units = {}, {}
    pending = []  # shapes with a label but no description of their own

    for el in root.iter():
        if el.tag.replace(SVG_NS, "") not in SHAPE_TAGS or DOMAIN_LABEL in (el.get(INKSCAPE_LABEL), el.get("id")):
            continue
        # nearest element (itself or a group above it) with a description
        node = el
        while node is not None and child_text(node, "desc") is None:
            node = parent.get(node)

        if node is None:
            if el.get(INKSCAPE_LABEL):
                pending.append(el)
            continue

        label = label_of(node)
        where = f"Shape '{label}' (id={node.get('id')})"
        rheology, props, thickness = parse_description(child_text(node, "desc"), where)
        unit = dict(rheology=rheology, props=props, thickness=thickness, title=child_text(node, "title"))
        if label in units and (units[label]["rheology"], units[label]["props"], units[label]["thickness"]) != (rheology, props, thickness):
            sys.exit(f"{where}: another shape with the same label has a different description.")
        units.setdefault(label, unit)
        shape_unit[el.get("id")] = label

    for el in pending:
        label = el.get(INKSCAPE_LABEL)
        if label in units:
            shape_unit[el.get("id")] = label
        else:
            print(f"Warning: shape '{label}' (id={el.get('id')}) has no description and was ignored.")

    if None in shape_unit:
        sys.exit("Some shapes with description have no id. Save the file with Inkscape to create them.")
    return shape_unit, units


def shape_to_mplpath(shape, n_curve=32):
    '''Convert an svgelements shape (already transformed) into a matplotlib Path.'''
    path = se.Path(shape)
    verts, codes = [], []
    for seg in path:
        if isinstance(seg, se.Move):
            verts.append((seg.end.x, seg.end.y))
            codes.append(MplPath.MOVETO)
        elif isinstance(seg, se.Close):
            verts.append((seg.end.x, seg.end.y))
            codes.append(MplPath.CLOSEPOLY)
        elif isinstance(seg, se.Line):
            verts.append((seg.end.x, seg.end.y))
            codes.append(MplPath.LINETO)
        else:  # curves and arcs: sample them
            for p in seg.npoint(np.linspace(0, 1, n_curve)[1:]):
                verts.append((p[0], p[1]))
                codes.append(MplPath.LINETO)
    if not verts:
        return None
    return MplPath(np.array(verts), codes)


def rasterize(svg_file, shape_unit, unit_names, Lx, Lz, Nx, dz, domain_id=None):
    '''
    Sample the SVG on a (Nzs, Nx) grid and return the unit index of each point (-1 = empty)
    and the unit it covers (the one that would be there without it, -1 = none).
    Also return the fill colour of each unit (from its first shape).
    '''
    svg = se.SVG.parse(svg_file, reify=True)

    # SVG region mapped to the domain: page, or the bounding box of an element
    if domain_id:
        ref = svg.get_element_by_id(domain_id)
        if ref is None:
            sys.exit(f"Element with id '{domain_id}' not found in the SVG.")
        x0, y0, x1, y1 = ref.bbox(with_stroke=False)
    else:
        x0, y0, x1, y1 = 0.0, 0.0, float(svg.width), float(svg.height)

    Nzs = int(round(Lz / dz))
    dz = Lz / Nzs
    x = np.linspace(0, Lx, Nx)
    zdepth = (np.arange(Nzs) + 0.5) * dz  # depth of samples (positive downwards)

    # sample points in SVG coordinates
    XS, YS = np.meshgrid(x0 + x / Lx * (x1 - x0), y0 + zdepth / Lz * (y1 - y0))
    pts = np.column_stack([XS.ravel(), YS.ravel()])

    grid = -np.ones(XS.shape, dtype=int)
    under = -np.ones(XS.shape, dtype=int)  # unit covered by the one painted at each point (-1 = none)
    colors = {}
    for el in svg.elements():
        if not isinstance(el, se.Shape) or el.id not in shape_unit:
            continue
        if el.values.get("display") == "none" or el.values.get("visibility") == "hidden":
            continue
        iunit = unit_names.index(shape_unit[el.id])
        if el.fill is not None and el.fill.value is not None:
            colors.setdefault(iunit, el.fill.hex[:7])
        mpath = shape_to_mplpath(el)
        if mpath is None:
            continue
        bx0, by0 = mpath.vertices.min(axis=0)
        bx1, by1 = mpath.vertices.max(axis=0)
        sel = (pts[:, 0] >= bx0) & (pts[:, 0] <= bx1) & (pts[:, 1] >= by0) & (pts[:, 1] <= by1)
        if not sel.any():
            continue
        inside = np.zeros(len(pts), dtype=bool)
        inside[sel] = mpath.contains_points(pts[sel])
        covered = inside & (grid.ravel() != iunit)
        under.ravel()[covered] = grid.ravel()[covered]
        grid.ravel()[inside] = iunit

    return x, zdepth, dz, grid, under, colors


def scs(a, b):
    '''Shortest common supersequence of two sequences.'''
    m, n = len(a), len(b)
    L = np.zeros((m + 1, n + 1), dtype=int)
    L[m, :] = np.arange(n, -1, -1)
    L[:, n] = np.arange(m, -1, -1)
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            L[i, j] = 1 + (L[i + 1, j + 1] if a[i] == b[j] else min(L[i + 1, j], L[i, j + 1]))
    out, i, j = [], 0, 0
    while i < m and j < n:
        if a[i] == b[j]:
            out.append(a[i]); i += 1; j += 1
        elif L[i + 1, j] <= L[i, j + 1]:
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    return out + list(a[i:]) + list(b[j:])


def embed(seq, stack):
    '''Slot index in stack of each element of seq (leftmost subsequence match).'''
    slots, j = [], 0
    for m in seq:
        while stack[j] != m:
            j += 1
        slots.append(j)
        j += 1
    return slots


def build_stack(grid, dz, Lz, unit_names):
    '''
    Return the stack of units (indices, bottom to top) and the base depth of each
    slot, an array (n_layers, Nx) in m (positive downwards).
    '''
    Nzs, Nx = grid.shape
    if (grid < 0).any():
        iz, ix = np.argwhere(grid < 0)[0]
        n = (grid < 0).sum()
        sys.exit(f"{n} sample points are not covered by any unit "
                 f"(first one at column {ix}, depth {(iz + 0.5) * dz / 1e3:.2f} km). "
                 "Check gaps between shapes or draw a background rectangle.")

    # runs of each column, from bottom to top: (units, depth of the top of each run)
    columns = []
    for ix in range(Nx):
        col = grid[::-1, ix]
        change = np.nonzero(np.diff(col))[0]  # last index of each run (except the top one)
        units = tuple(int(u) for u in col[np.r_[change, Nzs - 1]])
        tops = [Lz - (c + 1) * dz for c in change] + [0.0]
        columns.append((units, tops))

    unique = sorted({c[0] for c in columns}, key=len, reverse=True)
    stack = list(unique[0])
    for seq in unique[1:]:
        stack = scs(stack, seq)

    slots_of = {seq: embed(seq, stack) for seq in unique}
    base = np.empty((len(stack), Nx))
    base[0] = Lz
    for ix, (units, tops) in enumerate(columns):
        slots = slots_of[units]
        r = -1
        for k in range(1, len(stack)):
            # base of slot k = top of the highest run assigned to a slot below k
            while r + 1 < len(slots) and slots[r + 1] < k:
                r += 1
            base[k, ix] = tops[r] if r >= 0 else Lz
    return stack, unique, base


def rescale_thickness(stack, base, grid, under, unit_names, units, Lz, dz):
    '''
    Scale the drawing so that the units with h get that reference thickness.

    Each point of the drawing gets a vertical scale factor and, in each column,
    the depth of the interfaces is integrated from the top:
      - layers (units that reach both sides of the domain) with h: h / reference, where the
        reference is their most common drawn thickness (where they are not
        thinned or thickened). Thinned regions keep their drawn proportion;
      - layers without h: 1 (drawn thickness);
      - bodies (units that do not reach both sides: seeds, basins, weak
        zones, ...) without h: the factor of the unit they are drawn over (host);
      - bodies with h: h / reference, and their host in the same column absorbs
        the difference, so the units below do not move.
    The bottom unit fills the rest of the domain, so it can not have h.

    Return the new base depths (n_layers, Nx) and a list of messages describing the scaling.
    '''
    Nzs, Nx = grid.shape
    n_units = len(unit_names)
    has_h = np.array([units[name]["thickness"] is not None for name in unit_names])
    present = np.array([(grid == u).any(axis=0) for u in range(n_units)])  # (n_units, Nx)
    is_body = present.any(axis=1) & ~(present[:, 0] & present[:, -1])  # layers reach both sides of the domain

    def host_of(u):
        '''Unit most covered by u'''
        hosts = under[(grid == u) & (under >= 0)]
        return int(np.bincount(hosts).argmax()) if hosts.size else -1

    # factor of the units with h
    factor = np.ones(n_units)
    report = []
    for u, name in enumerate(unit_names):
        if not has_h[u] or not present[u].any():
            continue
        if u == stack[0]:
            sys.exit(f"Unit '{name}' is the bottom layer and fills the rest of the domain: remove h from its description.")
        thickness = (grid == u).sum(axis=0)[present[u]]  # drawn thickness in samples, where present
        values, counts = np.unique(thickness, return_counts=True)
        reference = values[np.argmax(counts)] * dz
        share = counts.max() / present[u].sum()
        if share < 0.2:
            print(f"Warning: unit '{name}' has no clear constant thickness (the most common one, "
                  f"{reference / 1e3:.1f} km, occurs in {100 * share:.0f}% of its columns). Check the scaling to h.")
        factor[u] = units[name]["thickness"] / reference
        kind = "body" if is_body[u] else "layer"
        report.append(f"{kind} '{name}': drawn {reference / 1e3:.1f} km (in {100 * share:.0f}% of its columns) "
                      f"-> h = {units[name]['thickness'] / 1e3:.1f} km (factor {factor[u]:.3f})")

    # effective factor of each unit: bodies without h follow their host
    def effective_factor(u, depth=0):
        if u < 0:
            return 1.0
        if has_h[u] or not is_body[u] or depth > n_units:
            return factor[u]
        return effective_factor(host_of(u), depth + 1)
    effective = np.array([effective_factor(u) for u in range(n_units)])

    def host_factor(mask):
        '''Factor of the units covered at the points of mask'''
        hosts = under[mask]
        return np.where(hosts >= 0, effective[np.maximum(hosts, 0)], 1.0)

    F = np.ones((Nzs, Nx))
    for u, name in enumerate(unit_names):
        mask = grid == u
        if is_body[u] and not has_h[u]:
            F[mask] = host_factor(mask)
            if mask.any() and (factor != 1).any():
                host = host_of(u)
                report.append(f"body '{name}': no h, follows the scaling of '{unit_names[host] if host >= 0 else '-'}'")
        else:
            F[mask] = factor[u]

    # bodies with h: their host absorbs, in each column, the difference to the host scaling
    for u, name in enumerate(unit_names):
        host = host_of(u)
        if not (is_body[u] and has_h[u]) or host < 0:
            continue
        mask = grid == u
        expected = np.zeros((Nzs, Nx))
        expected[mask] = host_factor(mask)
        diff = (factor[u] * mask.sum(axis=0) - expected.sum(axis=0)) * dz  # extra thickness of the body (m)
        host_mask = grid == host
        host_total = (F * host_mask).sum(axis=0) * dz
        cols = mask.any(axis=0) & (diff != 0)
        scale = np.ones(Nx)
        absorbed = cols & (host_total > 0)
        scale[absorbed] = (host_total[absorbed] - diff[absorbed]) / host_total[absorbed]
        n_short = np.sum(cols & ((host_total <= 0) | (scale < 0)))
        if n_short:
            print(f"Warning: in {n_short} column(s) '{unit_names[host]}' is too thin to absorb the thickness of "
                  f"'{name}' (h); the units below are pushed down there.")
        F[host_mask] *= np.broadcast_to(np.clip(scale, 0.0, None), (Nzs, Nx))[host_mask]

    # new depth of the boundaries between samples, integrated from the top
    depth_new = np.vstack([np.zeros((1, Nx)), np.cumsum(F * dz, axis=0)])
    new_base = np.take_along_axis(depth_new, np.round(base / dz).astype(int), axis=0)
    new_base[0] = Lz

    if new_base[1].max() > Lz:
        ix = int(np.argmax(new_base[1]))
        sys.exit(f"The layers above the bottom unit are {new_base[1, ix] / 1e3:.1f} km thick at column {ix}, "
                 f"more than Lz = {Lz / 1e3:.1f} km. Reduce h of some units.")
    return new_base, report


def layers_from_svg(svg_file, Lx, Lz, Nx, dz=100.0, domain_id=None, verbose=True):
    '''
    Read the SVG and return the list of MandyocLayer (bottom to top) with their
    interfaces (base depth, m) set. Lx, Lz and dz in m.
    The region mapped to the domain is the shape labelled "domain" if there is one,
    the element domain_id if given, or else the page.

    Parameters
    ----------
    svg_file : str
        Path to the SVG file.
    Lx : float
        Horizontal length of the domain in meters.
    Lz : float
        Vertical length of the domain in meters.
    Nx : int
        Number of horizontal grid points.
    dz : float, optional
        Vertical grid spacing in meters. Default is 100.0.
    domain_id : str, optional
        ID of the SVG element to use as the domain. If None, the shape labelled "domain" is used if present, otherwise the entire page is used. Default is None.
    verbose : bool, optional
        If True, print information about the layers and scaling. Default is True.
    '''
    if domain_id is None:
        domain = read_domain(svg_file)
        if domain is not None:
            domain_id = domain["id"]
    shape_unit, units = read_units(svg_file)
    if not units:
        sys.exit("No shape with description found in the SVG.")
    unit_names = list(units)

    x, zdepth, dz, grid, under, colors = rasterize(svg_file, shape_unit, unit_names, Lx, Lz, Nx, dz, domain_id)
    stack, unique, base = build_stack(grid, dz, Lz, unit_names)

    # consistency check: rebuild the unit field of the drawing from the interfaces (Mandyoc convention)
    rebuilt = np.array(stack)[(zdepth[:, None, None] < base[None, 1:, :].transpose(0, 2, 1)).sum(axis=2)]
    mismatch = (rebuilt != grid).sum()

    # scale the units with a reference thickness h
    base, scaling = rescale_thickness(stack, base, grid, under, unit_names, units, Lz, dz)

    # one MandyocLayer per unit, copied for each slot of the stack
    unit_layers = []
    for i, name in enumerate(unit_names):
        u = units[name]
        props = dict(u["props"])
        layer = MandyocLayer(name, u["rheology"],
                             density=props.pop("density"),
                             effective_viscosity_scale_factor=props.pop("effective_viscosity_scale_factor"),
                             radiogenic_heat_production=props.pop("radiogenic_heat_production"),
                             Nx=Nx)
        for attr, value in props.items():  # A, n, Q, V given in the description
            setattr(layer, attr, value)
        layer.title = u["title"] or name
        layer.color = colors.get(i, "#808080")
        layer.reference_thickness = u["thickness"]
        unit_layers.append(layer)

    layers = []
    for k, iunit in enumerate(stack):
        layer = copy.copy(unit_layers[iunit])
        layer.interface = base[k].copy()
        layer.base_depth = base[k].max()
        layers.append(layer)

    if verbose:
        for line in scaling:
            print(f"Thickness scaling: {line}")
        print(f"Distinct column patterns: {len(unique)}")
        print("Layers (bottom to top):")
        for k, layer in enumerate(layers):
            print(f"  {k:2d}  {layer.layer_label:25s} {layer.rheology_name:12s} "
                  f"C={layer.effective_viscosity_scale_factor:g} rho={layer.density:g} H={layer.radiogenic_heat_production:.4g}")
        print(f"n_interfaces = {len(layers) - 1}   # put this in param.txt")
        print(f"Sampling: dx = {x[1] - x[0]:.1f} m, dz = {dz:.1f} m; mismatched samples of the drawing after rebuild: {mismatch}")

    return layers


def write_interfaces(layers, fname="interfaces.txt"):
    '''Write interfaces.txt as in generate_input_files*.py.'''
    to_save = []
    for symbol, prop in zip(rheological_symbols, rheological_properties):
        to_save.append(f"{symbol} {' '.join([str(layer.__dict__[prop]) for layer in layers])}")

    with open(fname, "w") as f:
        for line in to_save:
            if len(line):
                f.write(f"{' '.join(line.split())}\n")

        # layer interfaces
        data = -1 * np.array(tuple(layer.interface for layer in layers[1::])).T
        np.savetxt(f, data, fmt="%.1f")


def plot_layers(layers, Lx, Lz, fname):
    x = np.linspace(0, Lx, layers[0].Nx)
    fig, ax = plt.subplots(figsize=(14, 6))
    for k, layer in enumerate(layers):
        top = layers[k + 1].interface if k < len(layers) - 1 else np.zeros_like(x)
        ax.fill_between(x / 1e3, -layer.interface / 1e3, -top / 1e3, color=layer.color, lw=0)
    ax.set_xlim(0, Lx / 1e3)
    ax.set_ylim(-Lz / 1e3, 0)
    ax.set_xlabel("Distance [km]")
    ax.set_ylabel("z [km]")

    # one legend entry per unit, from top to bottom
    unique = {}
    for layer in layers[::-1]:
        unique.setdefault(layer.layer_label, layer)
    handles = [Patch(facecolor=layer.color, edgecolor="k",
                     label=f"{layer.title}\n{layer.effective_viscosity_scale_factor:g} x {layer.rheology_name.replace('_', ' ')}\n"
                           fr"$\rho$ = {layer.density:g} kg/m³")
               for layer in unique.values()]
    fig.subplots_adjust(bottom=0.28)
    ax.legend(handles=handles, ncol=len(handles), loc="lower center", bbox_to_anchor=(0.5, -0.42),
              frameon=False, title="Layers properties", title_fontsize=12, fontsize=8)
    fig.savefig(fname, dpi=200)
    plt.close(fig)


###############################################################################
# Main
###############################################################################

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("svg", help="SVG file with the scenario")
    parser.add_argument("--Lx", type=float, help="domain width [km] (default: from the domain shape)")
    parser.add_argument("--Lz", type=float, help="domain depth [km] (default: from the domain shape)")
    parser.add_argument("--dx", type=float, help="horizontal resolution [km] (default: from the domain shape)")
    parser.add_argument("--Nx", type=int, help="number of points in x (alternative to --dx)")
    parser.add_argument("--sampling", type=float, default=0.1, help="vertical sampling to locate the interfaces [km] (default 0.1)")
    parser.add_argument("--domain-id", help="id of an element whose bounding box is the domain (default: the shape labelled 'domain', or the page)")
    parser.add_argument("-o", "--output", default="interfaces.txt")
    parser.add_argument("--plot", help="save a figure of the resulting layers (e.g. numerical_setup.png)")
    args = parser.parse_args()

    domain = read_domain(args.svg) or {}
    Lx = args.Lx * 1.0e3 if args.Lx else domain.get("Lx")
    Lz = args.Lz * 1.0e3 if args.Lz else domain.get("Lz")
    if Lx is None or Lz is None:
        sys.exit(f"No shape labelled '{DOMAIN_LABEL}' in the SVG: give --Lx and --Lz (and --dx or --Nx).")
    if args.dx:
        Nx = points_from_spacing(Lx, args.dx * 1.0e3, "dx")
    else:
        Nx = args.Nx or domain.get("Nx")
    if Nx is None:
        sys.exit("Give the horizontal resolution with --dx or --Nx.")
    print(f"Domain: Lx = {Lx / 1e3:g} km, Lz = {Lz / 1e3:g} km, Nx = {Nx}")

    layers = layers_from_svg(args.svg, Lx, Lz, Nx, dz=args.sampling * 1.0e3, domain_id=args.domain_id)
    write_interfaces(layers, args.output)
    print(f"Wrote {args.output}")

    if args.plot:
        plot_layers(layers, Lx, Lz, args.plot)
        print(f"Wrote {args.plot}")


if __name__ == "__main__":
    main()

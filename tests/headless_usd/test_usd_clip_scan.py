"""``simul.usd.clips`` finds every prim spec authoring value clips, variants included.

The composition sandbox and the animation range both use it; it replaced a
walk over spec objects that took 4 s on a 200,000-prim layer.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pxr", reason="pxr library not available")

from pxr import Sdf  # noqa: E402

from simul.usd.clips import clip_sets, clip_spec_paths  # noqa: E402

LAYER = """#usda 1.0
def "Plain" {
    double x = 1
    def "Child" {}
}
def "Top" (
    clips = {
        dictionary a = {
            asset[] assetPaths = [@./clip.usda@]
            string primPath = "/M"
        }
    }
) {
    def "Nested" (
        clips = {
            dictionary b = {
                asset manifestAssetPath = @./manifest.usda@
                string primPath = "/M"
            }
        }
    ) {}
}
def "V" (
    variants = { string look = "a" }
    prepend variantSets = "look"
) {
    variantSet "look" = {
        "a" {
            def "Inner" (
                clips = {
                    dictionary c = {
                        string templateAssetPath = "./clip.###.usda"
                        string primPath = "/M"
                    }
                }
            ) {}
        }
        "b" {}
    }
}
"""


def test_every_clip_spec_is_found_in_every_variant() -> None:
    layer = Sdf.Layer.CreateAnonymous(".usda")
    layer.ImportFromString(LAYER)

    paths = sorted(str(path) for path in clip_spec_paths(layer))

    assert paths == ["/Top", "/Top/Nested", "/V{look=a}Inner"]
    sets = {str(path): sorted(found) for path, found in clip_sets(layer)}
    assert sets == {"/Top": ["a"], "/Top/Nested": ["b"], "/V{look=a}Inner": ["c"]}


def test_a_layer_without_clips_has_none() -> None:
    layer = Sdf.Layer.CreateAnonymous(".usda")
    layer.ImportFromString('#usda 1.0\ndef "A" {\n    def "B" {}\n}\n')

    assert clip_spec_paths(layer) == []

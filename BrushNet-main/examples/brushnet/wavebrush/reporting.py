"""Exact learned-parameter counts; buffers such as RMS are excluded."""
def count(module):
    return sum(p.numel() for p in module.parameters()) if module is not None else 0


def parameter_report(wave, brushnet=None):
    out = {'brushnet': count(brushnet), 'wave': count(wave)}
    out['trainable'] = sum(p.numel() for m in (brushnet, wave) if m is not None for p in m.parameters() if p.requires_grad)
    if wave is not None:
        out.update(adapters=count(wave.adapters), projections=count(wave.projections), gates=count(wave.gates), zero_convs=count(wave.zero))
        out['adapter_each'] = [count(m) for m in wave.adapters]
        out['config'] = wave.config
        active = list(getattr(wave, 'active_slot_indices', range(len(wave.spec['slots']))))
        layer_by_slot = {slot: layer for slot, layer in zip(active, wave.zero)}
        out['active_slot_indices'] = active
        out['inactive_slot_indices'] = [i for i in range(len(wave.spec['slots'])) if i not in layer_by_slot]
        out['slots'] = [
            dict(
                slot=i,
                active=i in layer_by_slot,
                **slot_spec,
                parameters=count(layer_by_slot.get(i)),
            )
            for i, slot_spec in enumerate(wave.spec['slots'])
        ]
        out['wave_to_brushnet_percent'] = 100 * out['wave'] / out['brushnet'] if out['brushnet'] else None
    return out


def standard_sd15_spec():
    # Reference only: forward probe of the actual checkpoint takes precedence.
    down = [(320,0)] + [(320,0)]*2 + [(320,1)] + [(640,1)]*2 + [(640,2)] + [(1280,2)]*2 + [(1280,3)] + [(1280,3)]*2
    up = [(1280,3)]*3 + [(1280,2)] + [(1280,2)]*3 + [(1280,1)] + [(640,1)]*3 + [(640,0)] + [(320,0)]*3
    return {'slots':[{'channels':c,'scale':s} for c,s in down+[(1280,3)]+up], 'n_down':len(down), 'n_up':len(up)}

def get_steps_for_sliding_window(shape, patch_size, overlap):
    """Stub for mmunetvae predict."""
    steps = []
    for i in range(len(shape)):
        step = max(1, int(patch_size[i] * (1 - overlap)))
        max_start = max(0, shape[i] - patch_size[i])
        st = list(range(0, max_start + 1, step))
        if st and st[-1] != max_start:
            st.append(max_start)
        steps.append(st)
    return steps

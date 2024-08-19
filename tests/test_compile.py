from tensora_cffi.compile import compile_evaluate, tensor_cdefs


def test_compile():
    # tensora 'a(i) = b(i) + c(i)' -f a:s -f b:s -f c:s -t evaluate
    source = """
    int32_t evaluate(taco_tensor_t* restrict a, taco_tensor_t* restrict b, taco_tensor_t* restrict c) {
    // Extract dimensions
    int32_t i_dim = a->dimensions[0];
    
    // Unpack tensors
    int32_t* restrict a_0_pos = a->indices[0][0];
    int32_t* restrict a_0_crd = a->indices[0][1];
    double* restrict a_vals = a->vals;
    int32_t* restrict b_0_pos = b->indices[0][0];
    int32_t* restrict b_0_crd = b->indices[0][1];
    double* restrict b_vals = b->vals;
    int32_t* restrict c_0_pos = c->indices[0][0];
    int32_t* restrict c_0_crd = c->indices[0][1];
    double* restrict c_vals = c->vals;
    
    // Output initialization
    int32_t a_0_pos_capacity = 1 + 1;
    a_0_pos = malloc(sizeof(int32_t) * a_0_pos_capacity);
    a_0_pos[0] = 0;
    int32_t a_0_crd_capacity = 1024 * 1024;
    a_0_crd = malloc(sizeof(int32_t) * a_0_crd_capacity);
    int32_t p_0_a_0 = 0;
    int32_t a_vals_capacity = 1024 * 1024;
    a_vals = malloc(sizeof(double) * a_vals_capacity);
    
    // *** Iteration over i ***
    int32_t p_1_b_0 = b_0_pos[0];
    int32_t p_1_b_0_end = b_0_pos[1];
    int32_t p_2_c_0 = c_0_pos[0];
    int32_t p_2_c_0_end = c_0_pos[1];
    while (p_1_b_0 < p_1_b_0_end && p_2_c_0 < p_2_c_0_end) {
        int32_t i_1_b_0 = b_0_crd[p_1_b_0];
        int32_t i_2_c_0 = c_0_crd[p_2_c_0];
        int32_t i = TACO_MIN(i_1_b_0, i_2_c_0);
        if (i_1_b_0 == i && i_2_c_0 == i) {
        // vals allocation
        if (p_0_a_0 >= a_vals_capacity) {
            a_vals_capacity *= 2;
            a_vals = realloc(a_vals, sizeof(double) * a_vals_capacity);
        }
        
        // *** Computation of expression ***
        a_vals[p_0_a_0] = b_vals[p_1_b_0] + c_vals[p_2_c_0];
        
        // crd assembly
        if (p_0_a_0 >= a_0_crd_capacity) {
            a_0_crd_capacity *= 2;
            a_0_crd = realloc(a_0_crd, sizeof(int32_t) * a_0_crd_capacity);
        }
        a_0_crd[p_0_a_0] = i;
        
        p_0_a_0++;
        } else if (i_1_b_0 == i) {
        // vals allocation
        if (p_0_a_0 >= a_vals_capacity) {
            a_vals_capacity *= 2;
            a_vals = realloc(a_vals, sizeof(double) * a_vals_capacity);
        }
        
        // *** Computation of expression ***
        a_vals[p_0_a_0] = b_vals[p_1_b_0];
        
        // crd assembly
        if (p_0_a_0 >= a_0_crd_capacity) {
            a_0_crd_capacity *= 2;
            a_0_crd = realloc(a_0_crd, sizeof(int32_t) * a_0_crd_capacity);
        }
        a_0_crd[p_0_a_0] = i;
        
        p_0_a_0++;
        } else if (i_2_c_0 == i) {
        // vals allocation
        if (p_0_a_0 >= a_vals_capacity) {
            a_vals_capacity *= 2;
            a_vals = realloc(a_vals, sizeof(double) * a_vals_capacity);
        }
        
        // *** Computation of expression ***
        a_vals[p_0_a_0] = c_vals[p_2_c_0];
        
        // crd assembly
        if (p_0_a_0 >= a_0_crd_capacity) {
            a_0_crd_capacity *= 2;
            a_0_crd = realloc(a_0_crd, sizeof(int32_t) * a_0_crd_capacity);
        }
        a_0_crd[p_0_a_0] = i;
        
        p_0_a_0++;
        }
        p_1_b_0 += (int32_t)(i_1_b_0 == i);
        p_2_c_0 += (int32_t)(i_2_c_0 == i);
    }
    while (p_1_b_0 < p_1_b_0_end) {
        int32_t i_1_b_0 = b_0_crd[p_1_b_0];
        int32_t i = i_1_b_0;
        if (i_1_b_0 == i) {
        // vals allocation
        if (p_0_a_0 >= a_vals_capacity) {
            a_vals_capacity *= 2;
            a_vals = realloc(a_vals, sizeof(double) * a_vals_capacity);
        }
        
        // *** Computation of expression ***
        a_vals[p_0_a_0] = b_vals[p_1_b_0];
        
        // crd assembly
        if (p_0_a_0 >= a_0_crd_capacity) {
            a_0_crd_capacity *= 2;
            a_0_crd = realloc(a_0_crd, sizeof(int32_t) * a_0_crd_capacity);
        }
        a_0_crd[p_0_a_0] = i;
        
        p_0_a_0++;
        }
        p_1_b_0 += (int32_t)(i_1_b_0 == i);
    }
    while (p_2_c_0 < p_2_c_0_end) {
        int32_t i_2_c_0 = c_0_crd[p_2_c_0];
        int32_t i = i_2_c_0;
        if (i_2_c_0 == i) {
        // vals allocation
        if (p_0_a_0 >= a_vals_capacity) {
            a_vals_capacity *= 2;
            a_vals = realloc(a_vals, sizeof(double) * a_vals_capacity);
        }
        
        // *** Computation of expression ***
        a_vals[p_0_a_0] = c_vals[p_2_c_0];
        
        // crd assembly
        if (p_0_a_0 >= a_0_crd_capacity) {
            a_0_crd_capacity *= 2;
            a_0_crd = realloc(a_0_crd, sizeof(int32_t) * a_0_crd_capacity);
        }
        a_0_crd[p_0_a_0] = i;
        
        p_0_a_0++;
        }
        p_2_c_0 += (int32_t)(i_2_c_0 == i);
    }
    
    // pos assembly
    a_0_pos[1] = p_0_a_0;
    
    // Assembling output tensor a
    a->indices[0][0] = a_0_pos;
    a->indices[0][1] = a_0_crd;
    a->vals = a_vals;
    
    return 0;
    }"""

    lib = compile_evaluate(source)

    evaluate = lib.evaluate

    dimensions = tensor_cdefs.new("int32_t[]", [4])
    mode_ordering = tensor_cdefs.new("int32_t[]", [0])
    mode_types = tensor_cdefs.new("taco_mode_t[]", [1])
    empty_level = tensor_cdefs.new("int32_t*[]", [tensor_cdefs.NULL, tensor_cdefs.NULL])
    empty_indices = tensor_cdefs.new("int32_t**[]", [empty_level])

    a = tensor_cdefs.new(
        "taco_tensor_t*",
        {
            "order": 1,
            "dimensions": dimensions,
            "mode_ordering": mode_ordering,
            "mode_types": mode_types,
            "indices": empty_indices,
            "vals": tensor_cdefs.NULL,
        },
    )

    pos = tensor_cdefs.new("int32_t[]", [0, 2])
    idx = tensor_cdefs.new("int32_t[]", [0, 2])
    level = tensor_cdefs.new("int32_t*[]", [pos, idx])
    indices = tensor_cdefs.new("int32_t**[]", [level])
    vals = tensor_cdefs.new("double[]", [1.5, 2.5])

    b = tensor_cdefs.new(
        "taco_tensor_t*",
        {
            "order": 1,
            "dimensions": dimensions,
            "mode_ordering": mode_ordering,
            "mode_types": mode_types,
            "indices": indices,
            "vals": vals,
        },
    )

    evaluate(a, b, b)

    assert a.vals[0] == 3.0
    assert a.vals[1] == 5.0

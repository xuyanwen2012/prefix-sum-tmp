add_rules("mode.debug", "mode.release")

target("prefix-sum")
    set_kind("binary")
    add_includedirs("include")
    add_files("src/*.cu")
    add_cugencodes("native")

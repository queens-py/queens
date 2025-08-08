<!---
To add changes to the changelog create a tag where the message starts with 'change: ' the rest is
done automatically by a pipeline. Releases are added automatically. Changes in this file will
be overwritten!
-->

# Changelog

* [delete-input-file-workflow](https://github.com/queens-py/queens/commit/d0943420e9a6f3b3e4fe1cf35fafe19db1921833) (2025-08-04): Delete the input file workflow.

* [move-to-src-layout](https://github.com/queens-py/queens/commit/846b124b74f89187fff9391733cd7da5ac9e30dc) (2025-06-29): Refactor the QUEENS code from a flat layout to a src layout to allow separate packages.

* [relax-dependencies](https://github.com/queens-py/queens/commit/2b36e7fcb050658b7c7c711e3a065ccfad80b6fe) (2025-06-03): Remove fixed dependencies from the default installation and add a safe installation with fixed dependencies.

* [rename-simulation-files](https://github.com/queens-py/queens/commit/6fc58ce3056d724ba29426fa24c527fbb19e9fb9) (2025-05-15): Simplify the naming of files in the experiment directory.

* [simplify-experiment-directory-structure](https://github.com/queens-py/queens/commit/993974f76d186ea1a8909ee50852be2a9f95e343) (2025-05-14): Simplify the experiment directory structure.

* [add-reinforcement-learning](https://github.com/queens-py/queens/commit/867fec1549002b69be087c09629f76798c1f61d1) (2025-03-13): Add a reinforcement learning iterator and model.

* [introduce-naming-standard](https://github.com/queens-py/queens/commit/9a6fd0566d7417816877f329a9ddadb235ca00e0) (2025-02-26): Rename QUEENS classes and packages to conform to the new [QUEENS naming standard](https://github.com/queens-py/queens/issues/119) for files, modules, and classes.

* [add-multilevel-monte-carlo](https://github.com/queens-py/queens/commit/45f6f370d5a13aa6b13c0f534beadfcc8481df00) (2025-01-22): Add a multilevel Monte Carlo iterator and a control variates iterator.

* [allow-multiple-input-files](https://github.com/queens-py/queens/commit/b886962e69012246e47415e03170e2c9437390c1) (2024-10-09): Natively support multiple input files in jobscript driver

* [remove_interfaces_and_push_parameters_to_drivers](https://github.com/queens-py/queens/commit/c6fc37b9d5d6bc76cf7fc5492a147e26671def57) (2024-10-01): Remove interfaces and push parameters to drivers

* [remove-direct-python-interface](https://github.com/queens-py/queens/commit/0c911c4fd40549c147052e8110ca2a8478eb1ffb) (2024-09-05): Remove direct python interface and replace by pool scheduler and python driver

* [delete_old_random_field_implementation](https://github.com/queens-py/queens/commit/1dde8f0a4e6063a68ac7519502252403eab1fb67) (2024-05-21): Delete old random field implementation

* [dask_backend](https://github.com/queens-py/queens/commit/6d07402543432bf9a72c83535f6bf5de76881c8b) (2023-07-24): With this tag the 'old' workflow with interface-resources-scheduler-driver as well as the singularity dependency is removed. The simulation handling as well as remote communication is handled by dask.

* [delete_old_mf_modules](https://github.com/queens-py/queens/commit/a1dd60e20bd276516be93f18fc34bc90e3829140) (2023-03-29): Delete untested and deprecated multi-fidelity modules.

* [jinja_injection](https://github.com/queens-py/queens/commit/7e2cc701658e9739a397d6001cd32f4a44444673) (2023-01-31): QUEENS now uses jinja2 as injector back-end. Hence, the placeholder in the input templates needs double braces, e.g. `{{ place_holder_name  }}`.

* [add_yaml_input_support](https://github.com/queens-py/queens/commit/ab21ab029f8e0a7088d2c86e59eb841762a06828) (2022-07-21): Besides the json format, we now also support yaml format for the QUEENS input files.

* [external-python-modules](https://github.com/queens-py/queens/commit/2e2c726b978236a158100909733c7f099fc2fe6f) (2022-07-13): We now allow overwriting and overloading QUEENS modules and classes by external Python modules.

* [variable-class-rebuild](https://github.com/queens-py/queens/commit/acc3ae6a13f51a8e49f1fd908e53002e712867da) (2022-07-07): Variable class was rebuilt: modern and elegant!

## [v1.2](https://github.com/queens-py/queens/commit/206fcbe6200dac29e44d2243c4afc6ef2515f0c6) (2022-02-04)
New design of QUEENS post_post module

## [v1.1](https://github.com/queens-py/queens/commit/93a93661151cf09adc42f219c69d92749c93834d) (2022-01-03)
QUEENS is now installed using pip. Additionally, all required packages are pinned to a fixed version.

## [v1.0](https://github.com/queens-py/queens/commit/5c380cf7095e874e7670785d17ae7867e20a7982) (2021-11-04)
Release 1.0 (status prior transition)

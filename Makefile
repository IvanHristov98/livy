.PHONY: setup
setup: create-venv install-deps

.PHONY: create-venv
create-venv:
	$(call create_venv)

.PHONY: install-deps
install-deps:
	@pip install -r requirements.txt


.PHONY: run
run:
	@(cmd/utility/run_in_common_venv.sh "cmd" "dedup/dedup.py")

.PHONY: test-subsystem
test-subsystem:
	@(cmd/utility/run_in_common_venv.sh "." "-m" "unittest" "discover" "tests/subsystem")

.PHONY: test-unit
test-unit:
	@(cmd/utility/run_in_common_venv.sh "." "-m" "unittest" "discover" "tests/unit")

define create_venv
	@if [ ! -d ".venv" ]; then\
		python3 -m venv .venv;\
	fi
endef

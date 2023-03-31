.PHONY: setup
setup: create-venv install-deps

.PHONY: create-venv
create-venv:
	$(call create_venv)

.PHONY: install-deps
install-deps:
	@pip install -r requirements.txt


.PHONY: run-dedup
run-dedup:
	@(cmd/utility/run_in_common_venv.sh "cmd" "dedup/dedup.py")


define create_venv
	@if [ ! -d ".venv" ]; then\
		python3 -m venv .venv;\
	fi
endef

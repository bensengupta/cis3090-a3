clean:
	rm -f out.png out/*

zip:
	zip -r cis3090_a3.zip \
		README.md \
		a3.py \
		pyproject.toml \
		uv.lock \
		.python-version

# Docker stuff
container-start:
	docker run \
		--rm \
		--name cis3090-container \
		--volume .:/home/socs/app \
		--workdir /home/socs/app \
		--user socs \
		--detach \
		--tty \
		socsguelph/cis3090

container-stop:
	docker kill cis3090-container

container-connect:
	-docker exec --interactive --tty cis3090-container /bin/bash

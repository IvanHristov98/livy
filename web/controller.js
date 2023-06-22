function listenOnAddSubmit() {
    var form = document.getElementById("add-form");
    var alert = document.getElementById("add-alert")

    form.onsubmit = async (e) => {
        e.preventDefault();
        const form = e.currentTarget;
        const url = form.action;
      
        try {
            const formData = new FormData(form);
            const response = await fetch(url, {
                method: 'POST',
                body: formData
            });

            if (response.status == 200) {
                alert.style.visibility = "visible"
            }
        } catch (error) {
            console.error(error);
        }
    }
}

function listenOnTopNSubmit() {
    var form = document.getElementById("topn-form")
    var imsContainer = document.getElementById("top-n-ims")

    form.onsubmit = async (e) => {
        e.preventDefault();
        const form = e.currentTarget
        const url = form.action
      
        try {
            const formData = new FormData(form)
            const response = await fetch(url, {
                method: 'POST',
                body: formData
            })

            data = await response.json()

            imsContainer.innerHTML = ""

            for (let i = 0; i < data.im_ids.length; i++) {
                console.log(data.im_ids[i])
                visualizeImage(data.im_ids[i], imsContainer)
            }
        } catch (error) {
            console.error(error);
        }
    }
}

async function visualizeImage(imID, parentContainer) {
    const response = await fetch("http://localhost:8000/api/dedup/image/"+imID)
    const blob = await response.blob()
    const blobURL = URL.createObjectURL(blob)

    const col = document.createElement("div")
    col.classList.add("col-sm-6")
    col.classList.add("im-box")
    col.style.backgroundImage = "url(" + blobURL + ")"

    parentContainer.append(col)
}

listenOnAddSubmit()
listenOnTopNSubmit()